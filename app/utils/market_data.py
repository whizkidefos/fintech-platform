from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
from dataclasses import dataclass
from functools import lru_cache
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    vwap: Optional[float] = None

class MarketDataFetcher:
    def __init__(self, cache_timeout: int = 5):
        self.cache_timeout = cache_timeout
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=100)
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0.0),
                'change': info.get('regularMarketChange', 0.0),
                'change_percent': info.get('regularMarketChangePercent', 0.0),
                'volume': info.get('regularMarketVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'name': info.get('longName', symbol),
                'type': info.get('quoteType', 'EQUITY'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching ticker data for {symbol}: {str(e)}")
            raise

    def get_candles(self, symbol: str, timeframe: str = '1d',
                   limit: int = 100) -> List[Dict]:
        """Get historical candle data"""
        try:
            # Convert timeframe to yfinance interval
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d',
                '1w': '1wk',
                '1M': '1mo'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Calculate start date based on limit and interval
            end_date = datetime.now()
            if interval in ['1m', '5m', '15m', '30m']:
                start_date = end_date - timedelta(days=7)  # Yahoo limits intraday data
            else:
                start_date = end_date - timedelta(days=limit)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Convert to list of dictionaries
            candles = []
            for index, row in df.iterrows():
                candle = {
                    'timestamp': index.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'vwap': float(row['Close'])  # Simplified VWAP
                }
                candles.append(candle)
            
            return candles[-limit:]  # Return only requested number of candles
            
        except Exception as e:
            self.logger.error(
                f"Error fetching candle data for {symbol}: {str(e)}"
            )
            raise

    def get_market_depth(self, symbol: str) -> Dict:
        """Get market depth data (order book)"""
        try:
            # Note: Yahoo Finance doesn't provide order book data
            # This is a simplified version
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            current_price = info.get('regularMarketPrice', 0.0)
            spread = current_price * 0.001  # Simplified 0.1% spread
            
            return {
                'bids': [
                    {
                        'price': current_price - spread,
                        'quantity': 100
                    }
                ],
                'asks': [
                    {
                        'price': current_price + spread,
                        'quantity': 100
                    }
                ],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(
                f"Error fetching market depth for {symbol}: {str(e)}"
            )
            raise

    def get_market_summary(self) -> Dict:
        """Get overall market summary"""
        try:
            indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, Nasdaq
            summary = {}
            
            for index in indices:
                ticker = yf.Ticker(index)
                info = ticker.info
                
                summary[index] = {
                    'price': info.get('regularMarketPrice', 0.0),
                    'change': info.get('regularMarketChange', 0.0),
                    'change_percent': info.get('regularMarketChangePercent', 0.0),
                    'volume': info.get('regularMarketVolume', 0)
                }
            
            return {
                'indices': summary,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching market summary: {str(e)}")
            raise

    def get_company_info(self, symbol: str) -> Dict:
        """Get detailed company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'description': info.get('longBusinessSummary')
            }
        except Exception as e:
            self.logger.error(
                f"Error fetching company info for {symbol}: {str(e)}"
            )
            raise

    def get_technical_indicators(self, symbol: str) -> Dict:
        """Calculate technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1y')
            
            # Calculate indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'sma': {
                    '20': float(sma_20),
                    '50': float(sma_50),
                    '200': float(sma_200)
                },
                'rsi': float(rsi),
                'macd': {
                    'macd': float(macd.iloc[-1]),
                    'signal': float(signal.iloc[-1]),
                    'hist': float(macd.iloc[-1] - signal.iloc[-1])
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(
                f"Error calculating indicators for {symbol}: {str(e)}"
            )
            raise

    def clear_cache(self):
        """Clear the cache"""
        self.get_ticker.cache_clear()