[file name]: trading_env.py
[file content begin]
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class TradingEnvironment:
    def __init__(self, cfg: Dict = None, expert_ensemble=None):
        self.cfg = cfg or {}
        self.expert_ensemble = expert_ensemble
        self.trading_mode = self.cfg.get('trading_mode', 'spot')
        self.leverage = self.cfg.get('leverage', 10) if self.trading_mode == 'future' else 1
        self.fee_rate = self.cfg.get('fee_rate', 0.001) if self.trading_mode == 'spot' else self.cfg.get('fee_rate', 0.0004)
        self.slippage = self.cfg.get('slippage', 0.0005)
        self.initial_capital = self.cfg.get('initial_capital', 10000.0)
        self.max_risk_per_trade = self.cfg.get('max_risk_per_trade', 0.02)
        self.max_position_pct = self.cfg.get('max_position_pct', 0.5)
        self.drawdown_penalty = self.cfg.get('drawdown_penalty', 2.0)
        self.invalid_action_penalty = self.cfg.get('invalid_action_penalty', -0.1)
        self.max_hold_bars = 4
        self.tp_pct = self.cfg.get('take_profit_pct', 0.003)
        self.sl_pct = self.cfg.get('stop_loss_pct', 0.0015)
        self.use_trailing = self.cfg.get('use_trailing', True)
        self.trailing_activation_pct = self.cfg.get('trailing_activation_pct', 0.015)
        self.trailing_callback_pct = self.cfg.get('trailing_callback_pct', 0.01)
        self.df = None
        self.features = None
        self.close_idx = 0
        self.window = self.cfg.get('window', 120)
        self.n_bars = 0
        self.current_idx = 0
        self.capital = self.initial_capital
        self.margin_locked = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.position_side = 'long'
        self.peak_capital = self.initial_capital
        self.done = False
        self.trades = []
        self.returns_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.invalid_scaling_count = 0
        self.dynamic_sl = 0.0
        self.dynamic_tp = 0.0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        self.bars_held = 0

    def reset(self, df: pd.DataFrame = None, scaled_features: np.ndarray = None, close_idx: int = None) -> np.ndarray:
        if df is not None:
            self.df = df.reset_index(drop=True)
            self.features = scaled_features
            self.close_idx = close_idx or 0
            self.n_bars = len(df)
        self.current_idx = self.window
        self.capital = self.initial_capital
        self.margin_locked = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.position_side = 'long'
        self.peak_capital = self.initial_capital
        self.done = False
        self.trades = []
        self.returns_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.invalid_scaling_count = 0
        self.dynamic_sl = 0.0
        self.dynamic_tp = 0.0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        self.bars_held = 0
        return self._get_ppo_state()

    def _get_expert_signals(self) -> Dict[str, float]:
        idx = min(self.current_idx, self.n_bars - 1)
        if self.expert_ensemble is not None:
            try:
                ensemble_features = [
                    'hurst_exp', 'efficiency_ratio_20', 'natr',
                    'rejection_high', 'rejection_low', 'range_percentile',
                    'price_accel', 'vol_aggression', 'stop_buy_dist',
                    'stop_sell_dist', 'vwap_ema_spread'
                ]
                row = self.df.iloc[idx]
                X = row[ensemble_features].values.astype(np.float32).reshape(1, -1)
                signals = self.expert_ensemble.predict_expert_signals(X)
                return {
                    'long_prob': float(signals.get('long_prob', 0.5)[0]) if hasattr(signals.get('long_prob'), '__len__') else float(signals.get('long_prob', 0.5)),
                    'short_prob': float(signals.get('short_prob', 0.5)[0]) if hasattr(signals.get('short_prob'), '__len__') else float(signals.get('short_prob', 0.5)),
                    'expected_return': float(signals.get('expected_return', 0.0)[0]) if hasattr(signals.get('expected_return'), '__len__') else float(signals.get('expected_return', 0.0)),
                    'uncertainty': float(signals.get('uncertainty', 0.01)[0]) if hasattr(signals.get('uncertainty'), '__len__') else float(signals.get('uncertainty', 0.01))
                }
            except Exception:
                pass
        return {
            'long_prob': 0.5,
            'short_prob': 0.5,
            'expected_return': 0.0,
            'uncertainty': 0.01
        }

    def _get_regime_probs(self) -> np.ndarray:
        regime_cols = ['regime_p_0', 'regime_p_1', 'regime_p_2']
        idx = min(self.current_idx, self.n_bars - 1)
        if all(col in self.df.columns for col in regime_cols) and idx < len(self.df):
            return self.df[regime_cols].iloc[idx].values.astype(np.float32)
        return np.array([0.33, 0.33, 0.34], dtype=np.float32)

    def _get_portfolio_status(self) -> Tuple[float, float, float]:
        if self.position == 0:
            position_status, pnl_pct = 0.0, 0.0
        else:
            position_status = 1.0 if self.position_side == 'long' else -1.0
            price = self._current_price()
            pnl_pct = (price - self.entry_price) / self.entry_price if self.position_side == 'long' else (self.entry_price - price) / self.entry_price
            if self.trading_mode == 'future':
                pnl_pct *= self.leverage
        val = self.get_portfolio_value()
        available_margin = (self.capital / val) if val > 0 else 1.0
        return position_status, float(pnl_pct), float(available_margin)

    def _get_ppo_state(self) -> np.ndarray:
        idx = min(self.current_idx, self.n_bars - 1)
        row = self.df.iloc[idx]
        expert = self._get_expert_signals()
        regime_probs = self._get_regime_probs()
        pos_status, pnl_pct, margin_ratio = self._get_portfolio_status()
        vpin = float(row['vpin']) if 'vpin' in row else 0.0
        vol_agg = float(row['vol_aggression']) if 'vol_aggression' in row else 0.0
        return np.array([
            expert['long_prob'],
            expert['short_prob'],
            expert['expected_return'],
            expert['uncertainty'],
            regime_probs[0],
            regime_probs[1],
            regime_probs[2],
            pos_status,
            pnl_pct,
            margin_ratio,
            vpin,
            vol_agg
        ], dtype=np.float32)

    def _calculate_reward(self, pnl_pct: float, reason: str = 'none') -> float:
        if reason == 'tp':
            return 1.0
        elif reason == 'sl':
            return -1.5
        elif reason == 'timeout':
            return pnl_pct * 10.0
        else:
            if self.position != 0:
                return -0.005
            return 0.0

    def _current_price(self) -> float:
        return float(self.df['close'].iloc[min(self.current_idx, self.n_bars - 1)])

    def _current_high(self) -> float:
        return float(self.df['high'].iloc[min(self.current_idx, self.n_bars - 1)])

    def _current_low(self) -> float:
        return float(self.df['low'].iloc[min(self.current_idx, self.n_bars - 1)])

    def get_portfolio_value(self) -> float:
        price = self._current_price()
        if self.position == 0:
            return self.capital
        if self.trading_mode == 'spot':
            return self.capital + (self.position * price)
        unrealized = self.position * (price - self.entry_price) if self.position_side == 'long' else self.position * (self.entry_price - price)
        return self.capital + self.margin_locked + unrealized

    def _force_close_position(self, exit_price: float, reason: str) -> Tuple[float, str]:
        orig_margin = self.margin_locked
        if self.position_side == 'long':
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - exit_price) / self.entry_price
        if self.trading_mode == 'future':
            realized = self.position * (exit_price - self.entry_price) if self.position_side == 'long' else self.position * (self.entry_price - exit_price)
            fee = (self.position * exit_price) * self.fee_rate
            self.capital += (self.margin_locked + realized - fee)
            trade_pnl = realized / (orig_margin + 1e-10)
        else:
            gross = self.position * exit_price
            fee = gross * self.fee_rate
            self.capital += (gross - fee)
            trade_pnl = pnl_pct
        self.trades.append({
            'entry_idx': self.entry_idx, 'exit_idx': self.current_idx,
            'entry_price': self.entry_price, 'exit_price': exit_price,
            'pnl_pct': trade_pnl, 'side': self.position_side,
            'bars_held': self.current_idx - self.entry_idx, 'trigger': reason
        })
        if trade_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.margin_locked = 0.0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        return trade_pnl, reason

    def _execute_order(self, action: int) -> Tuple[bool, float, str]:
        price = self._current_price()
        if action == 0:
            return True, 0.0, 'hold'
        if action in [1, 2, 3]:
            size_pct, action_type = [0.25, 0.50, 1.0][action - 1], 'buy'
        elif action in [4, 5, 6]:
            size_pct, action_type = [0.25, 0.50, 1.0][action - 4], 'sell'
        else:
            action_type = 'close_all'
        if self.position == 0:
            if action_type == 'close_all':
                return True, 0.0, 'hold'
            trade_value = self.capital * size_pct
            if action_type == 'buy':
                entry_price = price * (1 + self.slippage)
                size = trade_value / entry_price
                if self.trading_mode == 'future':
                    margin, fee = trade_value / self.leverage, trade_value * self.fee_rate
                    if margin + fee <= self.capital:
                        self.capital -= (margin + fee)
                        self.margin_locked = margin
                        self.position = size
                        self.entry_price = entry_price
                        self.entry_idx = self.current_idx
                        self.position_side = 'long'
                        self.dynamic_sl = entry_price * (1 - self.sl_pct)
                        self.dynamic_tp = entry_price * (1 + self.tp_pct)
                        self.trailing_peak = entry_price
                        self.bars_held = 0
                        return True, size, 'buy_long_open'
                else:
                    cost = trade_value * (1 + self.fee_rate)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = size
                        self.entry_price = entry_price
                        self.entry_idx = self.current_idx
                        self.position_side = 'long'
                        self.dynamic_sl = entry_price * (1 - self.sl_pct)
                        self.dynamic_tp = entry_price * (1 + self.tp_pct)
                        self.bars_held = 0
                        return True, size, 'spot_buy_open'
            elif action_type == 'sell' and self.trading_mode == 'future':
                entry_price = price * (1 - self.slippage)
                size = trade_value / entry_price
                margin, fee = trade_value / self.leverage, trade_value * self.fee_rate
                if margin + fee <= self.capital:
                    self.capital -= (margin + fee)
                    self.margin_locked = margin
                    self.position = size
                    self.entry_price = entry_price
                    self.entry_idx = self.current_idx
                    self.position_side = 'short'
                    self.dynamic_sl = entry_price * (1 + self.sl_pct)
                    self.dynamic_tp = entry_price * (1 - self.tp_pct)
                    self.trailing_peak = entry_price
                    self.bars_held = 0
                    return True, size, 'sell_short_open'
            return False, 0.0, 'insufficient_margin'
        if self.position > 0 and self.position_side == 'long':
            if action_type == 'buy':
                return False, 0.0, 'invalid_scaling'
            if action_type == 'sell' or action_type == 'close_all':
                sell_pct = 1.0 if action_type == 'close_all' else size_pct
                close_size = self.position * sell_pct
                exit_price = price * (1 - self.slippage)
                orig_margin = self.margin_locked
                pnl_pct = (exit_price - self.entry_price) / self.entry_price
                if self.trading_mode == 'future':
                    realized = close_size * (exit_price - self.entry_price)
                    fee = (close_size * exit_price) * self.fee_rate
                    self.capital += ((self.margin_locked * sell_pct) + realized - fee)
                    self.margin_locked -= (self.margin_locked * sell_pct)
                    self.position -= close_size
                    trade_pnl = realized / (orig_margin * sell_pct + 1e-10)
                else:
                    gross = close_size * exit_price
                    self.capital += (gross - (gross * self.fee_rate))
                    self.position -= close_size
                    trade_pnl = pnl_pct
                if self.position < 1e-10:
                    self.trades.append({
                        'entry_idx': self.entry_idx, 'exit_idx': self.current_idx,
                        'entry_price': self.entry_price, 'exit_price': exit_price,
                        'pnl_pct': trade_pnl, 'side': 'long',
                        'bars_held': self.current_idx - self.entry_idx, 'trigger': 'agent_action'
                    })
                    if trade_pnl > 0:
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.margin_locked = 0.0
                return True, close_size, 'long_reduced_or_closed'
        if self.position > 0 and self.position_side == 'short':
            if action_type == 'sell':
                return False, 0.0, 'invalid_scaling'
            if action_type == 'buy' or action_type == 'close_all':
                buy_pct = 1.0 if action_type == 'close_all' else size_pct
                close_size = self.position * buy_pct
                exit_price = price * (1 + self.slippage)
                orig_margin = self.margin_locked
                if self.trading_mode == 'future':
                    realized = close_size * (self.entry_price - exit_price)
                    fee = (close_size * exit_price) * self.fee_rate
                    self.capital += ((self.margin_locked * buy_pct) + realized - fee)
                    self.margin_locked -= (self.margin_locked * buy_pct)
                    self.position -= close_size
                    trade_pnl = realized / (orig_margin * buy_pct + 1e-10)
                    if self.position < 1e-10:
                        self.trades.append({
                            'entry_idx': self.entry_idx, 'exit_idx': self.current_idx,
                            'entry_price': self.entry_price, 'exit_price': exit_price,
                            'pnl_pct': trade_pnl, 'side': 'short',
                            'bars_held': self.current_idx - self.entry_idx, 'trigger': 'agent_action'
                        })
                        if trade_pnl > 0:
                            self.consecutive_wins += 1
                            self.consecutive_losses = 0
                        else:
                            self.consecutive_losses += 1
                            self.consecutive_wins = 0
                        self.position = 0.0
                        self.entry_price = 0.0
                        self.margin_locked = 0.0
                return True, close_size, 'short_reduced_or_closed'
        return False, 0.0, 'hold'

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.bars_held += 1
        reward = 0.0
        reward_computed = False
        trade_closed_this_step = False
        step_pnl = 0.0
        close_reason = 'none'
        if self.position > 0 and self.bars_held >= self.max_hold_bars:
            price = self._current_price()
            exit_price = price * (1 - self.slippage) if self.position_side == 'long' else price * (1 + self.slippage)
            step_pnl, close_reason = self._force_close_position(exit_price, 'timeout')
            reward = self._calculate_reward(step_pnl, close_reason)
            reward_computed = True
            trade_closed_this_step = True
            action = 0
        if self.position > 0 and self.use_trailing and not trade_closed_this_step:
            price = self._current_price()
            if self.position_side == 'long':
                if price > self.trailing_peak:
                    self.trailing_peak = price
                profit_pct = (price - self.entry_price) / self.entry_price
                if profit_pct >= self.trailing_activation_pct:
                    self.trailing_activated = True
                if self.trailing_activated:
                    new_sl = self.trailing_peak * (1 - self.trailing_callback_pct)
                    if new_sl > self.dynamic_sl:
                        self.dynamic_sl = new_sl
            elif self.position_side == 'short':
                if price < self.trailing_peak or self.trailing_peak == 0.0:
                    self.trailing_peak = price
                profit_pct = (self.entry_price - price) / self.entry_price
                if profit_pct >= self.trailing_activation_pct:
                    self.trailing_activated = True
                if self.trailing_activated:
                    new_sl = self.trailing_peak * (1 + self.trailing_callback_pct)
                    if new_sl < self.dynamic_sl or self.dynamic_sl == 0.0:
                        self.dynamic_sl = new_sl
        if self.position > 0 and not trade_closed_this_step:
            high, low = self._current_high(), self._current_low()
            stopped_out = False
            exit_price, log_msg = 0.0, ''
            if self.position_side == 'long':
                if low <= self.dynamic_sl:
                    stopped_out = True
                    exit_price = self.dynamic_sl * (1 - self.slippage)
                    log_msg = 'sl'
                elif high >= self.dynamic_tp:
                    stopped_out = True
                    exit_price = self.dynamic_tp * (1 + self.slippage)
                    log_msg = 'tp'
            elif self.position_side == 'short':
                if high >= self.dynamic_sl:
                    stopped_out = True
                    exit_price = self.dynamic_sl * (1 + self.slippage)
                    log_msg = 'sl'
                elif low <= self.dynamic_tp:
                    stopped_out = True
                    exit_price = self.dynamic_tp * (1 - self.slippage)
                    log_msg = 'tp'
            if stopped_out:
                step_pnl, close_reason = self._force_close_position(exit_price, log_msg)
                reward = self._calculate_reward(step_pnl, close_reason)
                reward_computed = True
                trade_closed_this_step = True
                action = 0
        if not trade_closed_this_step:
            success, _, order_status = self._execute_order(action)
            if order_status == 'invalid_scaling':
                reward = self.invalid_action_penalty
                reward_computed = True
                self.invalid_scaling_count += 1
            elif order_status in ['long_reduced_or_closed', 'short_reduced_or_closed'] and self.position == 0:
                trade_closed_this_step = True
                if self.trades:
                    step_pnl = self.trades[-1]['pnl_pct']
                    close_reason = 'agent_action'
        if self.current_idx >= self.n_bars - 1:
            self.done = True
            if self.position > 0:
                price = self._current_price()
                exit_price = price * (1 - self.slippage) if self.position_side == 'long' else price * (1 + self.slippage)
                step_pnl, close_reason = self._force_close_position(exit_price, 'episode_end')
                reward = self._calculate_reward(step_pnl, close_reason)
                reward_computed = True
        if not reward_computed:
            if trade_closed_this_step:
                if close_reason == 'none':
                    close_reason = 'agent_action'
                reward = self._calculate_reward(step_pnl, close_reason)
            else:
                reward = self._calculate_reward(0.0, 'none')
        self.current_idx += 1
        next_state = self._get_ppo_state()
        return next_state, float(reward), self.done

    def get_trade_statistics(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'avg_win': 0.0,
                'avg_loss': 0.0, 'profit_factor': 0.0, 'total_return': 0.0,
                'invalid_scaling_attempts': self.invalid_scaling_count
            }
        wins = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in self.trades if t['pnl_pct'] <= 0]
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(np.abs(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = sum(np.abs(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        return {
            'total_trades': total_trades, 'win_rate': win_rate,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital,
            'invalid_scaling_attempts': self.invalid_scaling_count
        }

    def set_expert_ensemble(self, expert_ensemble) -> None:
        self.expert_ensemble = expert_ensemble