from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from app.utils.signal_generator import SignalStrategy, RiskManager
from app.utils.portfolio_analytics import PortfolioAnalytics

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float,
                 commission: float = 0.001,
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()

    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.trade_history = []
        self.equity_curve = []

    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy: SignalStrategy,
                    risk_manager: RiskManager) -> Dict[str, Any]:
        """Run backtest with given strategy and data"""
        self.reset()
        results = []
        
        for timestamp, candle in data.iterrows():
            # Update positions
            self._update_positions(candle)
            
            # Generate signals
            signals = strategy.generate_signals()
            if signals:
                for signal in signals:
                    self._execute_signal(signal, candle, risk_manager)
            
            # Record equity
            total_equity = self._calculate_total_equity(candle)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'cash': self.capital,
                'positions': self.positions.copy()
            })

        return self._generate_backtest_report()

    def _update_positions(self, candle: pd.Series):
        """Update open positions and check stop losses/take profits"""
        for symbol, position in list(self.positions.items()):
            # Check stop loss
            if position['direction'] == 'long':
                if candle['low'] <= position['stop_loss']:
                    self._close_position(symbol, position['stop_loss'], 'stop_loss', candle)
                elif candle['high'] >= position['take_profit']:
                    self._close_position(symbol, position['take_profit'], 'take_profit', candle)
            else:  # short position
                if candle['high'] >= position['stop_loss']:
                    self._close_position(symbol, position['stop_loss'], 'stop_loss', candle)
                elif candle['low'] <= position['take_profit']:
                    self._close_position(symbol, position['take_profit'], 'take_profit', candle)

    def _execute_signal(self, 
                       signal: Dict[str, Any],
                       candle: pd.Series,
                       risk_manager: RiskManager):
        """Execute trading signal"""
        symbol = signal['symbol']
        
        # Check if we already have a position
        if symbol in self.positions:
            if self.positions[symbol]['direction'] != signal['type']:
                # Close existing position if signal is in opposite direction
                self._close_position(symbol, candle['close'], 'signal_reversal', candle)
            else:
                return  # Skip if we already have a position in same direction
        
        # Calculate position size and risk parameters
        risk_params = risk_manager.calculate_position_size(candle['close'], signal['stop_loss'])
        
        # Apply slippage to entry price
        entry_price = candle['close'] * (1 + self.slippage if signal['type'] == 'buy' else 1 - self.slippage)
        
        # Calculate position size after commission
        position_size = risk_params['position_size'] * (1 - self.commission)
        
        # Check if we have enough capital
        required_capital = position_size * entry_price
        if required_capital > self.capital:
            position_size = (self.capital / entry_price) * 0.95  # Use 95% of available capital
        
        # Open new position
        self.positions[symbol] = {
            'direction': signal['type'],
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'entry_time': candle.name,
            'entry_signal': signal['strategy']
        }
        
        # Update capital
        self.capital -= required_capital * (1 + self.commission)
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'direction': signal['type'],
            'entry_time': candle.name,
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'signal_type': signal['strategy']
        })

    def _close_position(self, 
                       symbol: str,
                       exit_price: float,
                       exit_reason: str,
                       candle: pd.Series):
        """Close an open position"""
        position = self.positions[symbol]
        
        # Apply slippage to exit price
        actual_exit_price = exit_price * (1 - self.slippage if position['direction'] == 'long' else 1 + self.slippage)
        
        # Calculate PnL
        if position['direction'] == 'long':
            pnl = (actual_exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - actual_exit_price) * position['size']
        
        # Update capital
        self.capital += (position['size'] * actual_exit_price) * (1 - self.commission) + pnl
        
        # Record trade history
        self.trade_history.append({
            'symbol': symbol,
            'direction': position['direction'],
            'entry_time': position['entry_time'],
            'exit_time': candle.name,
            'entry_price': position['entry_price'],
            'exit_price': actual_exit_price,
            'position_size': position['size'],
            'pnl': pnl,
            'return_pct': (pnl / (position['entry_price'] * position['size'])) * 100,
            'exit_reason': exit_reason,
            'duration': (candle.name - position['entry_time']).total_seconds() / 3600,  # hours
            'entry_signal': position['entry_signal']
        })
        
        # Remove position
        del self.positions[symbol]

    def _calculate_total_equity(self, candle: pd.Series) -> float:
        """Calculate total account equity"""
        equity = self.capital
        for symbol, position in self.positions.items():
            current_price = candle['close']
            position_value = position['size'] * current_price
            equity += position_value
        return equity

    def _generate_backtest_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        equity_df = pd.DataFrame(self.equity_curve)
        trade_history_df = pd.DataFrame(self.trade_history)
        
        if len(trade_history_df) == 0:
            return {
                'error': 'No trades executed during backtest period'
            }

        # Calculate basic metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100
        equity_returns = equity_df['equity'].pct_change()
        
        # Risk metrics
        max_drawdown = ((equity_df['equity'] - equity_df['equity'].expanding().max()) / 
                       equity_df['equity'].expanding().max()).min() * 100
        
        sharpe_ratio = np.sqrt(252) * equity_returns.mean() / equity_returns.std()
        
        # Trade metrics
        win_rate = len(trade_history_df[trade_history_df['pnl'] > 0]) / len(trade_history_df) * 100
        avg_win = trade_history_df[trade_history_df['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(trade_history_df[trade_history_df['pnl'] < 0]['pnl'].mean())
        profit_factor = abs(trade_history_df[trade_history_df['pnl'] > 0]['pnl'].sum() / 
                          trade_history_df[trade_history_df['pnl'] < 0]['pnl'].sum())
        
        return {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_equity': equity_df['equity'].iloc[-1],
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trade_history_df),
                'avg_trade_return': trade_history_df['return_pct'].mean(),
                'avg_win_loss_ratio': avg_win / avg_loss if avg_loss != 0 else float('inf')
            },
            'trade_analysis': {
                'win_rate_by_signal': trade_history_df.groupby('entry_signal').apply(
                    lambda x: len(x[x['pnl'] > 0]) / len(x) * 100).to_dict(),
                'avg_duration': trade_history_df['duration'].mean(),
                'best_trade': trade_history_df.loc[trade_history_df['pnl'].idxmax()].to_dict(),
                'worst_trade': trade_history_df.loc[trade_history_df['pnl'].idxmin()].to_dict()
            },
            'equity_curve': equity_df.to_dict(orient='records'),
            'trade_history': trade_history_df.to_dict(orient='records'),
            'monthly_returns': equity_returns.resample('M').mean().to_dict()
        }