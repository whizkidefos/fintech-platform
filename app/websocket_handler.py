from flask_sock import Sock
import json
from datetime import datetime
from typing import Dict, Set
import asyncio
from app.utils.market_data import MarketDataFetcher
from app.models import Portfolio, Position, TradingSignal

class WebSocketManager:
    def __init__(self):
        self.clients: Dict[int, Set] = {}  # user_id -> set of websocket connections
        self.market_data = MarketDataFetcher()
        self.running = False
        self.update_tasks = set()

    def register_client(self, user_id: int, ws):
        """Register a new websocket connection for a user"""
        if user_id not in self.clients:
            self.clients[user_id] = set()
        self.clients[user_id].add(ws)

        if not self.running:
            self.start_updates()

    def unregister_client(self, user_id: int, ws):
        """Unregister a websocket connection"""
        if user_id in self.clients:
            self.clients[user_id].discard(ws)
            if not self.clients[user_id]:
                del self.clients[user_id]

        if not self.clients:
            self.stop_updates()

    async def send_to_user(self, user_id: int, message: dict):
        """Send message to all connections of a specific user"""
        if user_id in self.clients:
            dead_connections = set()
            for ws in self.clients[user_id]:
                try:
                    await ws.send(json.dumps(message))
                except Exception:
                    dead_connections.add(ws)

            # Clean up dead connections
            for ws in dead_connections:
                self.clients[user_id].discard(ws)

    def start_updates(self):
        """Start background update tasks"""
        self.running = True
        asyncio.create_task(self._portfolio_updates())
        asyncio.create_task(self._market_updates())
        asyncio.create_task(self._signal_updates())

    def stop_updates(self):
        """Stop all background update tasks"""
        self.running = False
        for task in self.update_tasks:
            task.cancel()
        self.update_tasks.clear()

    async def _portfolio_updates(self):
        """Send portfolio updates to users"""
        while self.running:
            try:
                for user_id in self.clients:
                    portfolio = Portfolio.query.filter_by(user_id=user_id).first()
                    if portfolio:
                        # Update portfolio values
                        portfolio.update_portfolio_value()
                        
                        # Send update to user
                        await self.send_to_user(user_id, {
                            'type': 'portfolio_update',
                            'data': {
                                'total_value': float(portfolio.equity_value + portfolio.cash_balance),
                                'equity_value': float(portfolio.equity_value),
                                'cash_balance': float(portfolio.cash_balance),
                                'unrealized_pnl': float(portfolio.unrealized_pnl),
                                'realized_pnl': float(portfolio.realized_pnl),
                                'timestamp': datetime.utcnow().isoformat()
                            }
                        })

                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Error in portfolio updates: {e}")
                await asyncio.sleep(5)

    async def _market_updates(self):
        """Send market data updates to users"""
        while self.running:
            try:
                for user_id in self.clients:
                    # Get watched assets for user
                    portfolio = Portfolio.query.filter_by(user_id=user_id).first()
                    if portfolio:
                        positions = Position.query.filter_by(
                            portfolio_id=portfolio.id,
                            status='open'
                        ).all()

                        for position in positions:
                            ticker = self.market_data.fetch_ticker(position.asset.symbol)
                            
                            await self.send_to_user(user_id, {
                                'type': 'price_update',
                                'data': {
                                    'symbol': position.asset.symbol,
                                    'price': ticker['last'],
                                    'change': ticker['percentage'],
                                    'volume': ticker['baseVolume'],
                                    'high': ticker['high'],
                                    'low': ticker['low'],
                                    'timestamp': datetime.utcnow().isoformat()
                                }
                            })

                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error in market updates: {e}")
                await asyncio.sleep(5)

    async def _signal_updates(self):
        """Send trading signal updates to users"""
        while self.running:
            try:
                for user_id in self.clients:
                    # Get recent signals
                    signals = TradingSignal.query.filter(
                        TradingSignal.created_at > datetime.utcnow()
                    ).all()

                    for signal in signals:
                        await self.send_to_user(user_id, {
                            'type': 'signal',
                            'data': {
                                'id': signal.id,
                                'asset': signal.asset.symbol,
                                'type': signal.signal_type,
                                'strength': signal.strength,
                                'indicators': signal.indicator_values,
                                'timestamp': signal.created_at.isoformat()
                            }
                        })

                await asyncio.sleep(60)  # Check for new signals every minute
            except Exception as e:
                print(f"Error in signal updates: {e}")
                await asyncio.sleep(5)

    async def handle_client_message(self, user_id: int, message: dict):
        """Handle incoming messages from clients"""
        try:
            message_type = message.get('type')
            data = message.get('data', {})

            if message_type == 'subscribe':
                # Handle subscription requests
                await self._handle_subscription(user_id, data)
            elif message_type == 'unsubscribe':
                # Handle unsubscription requests
                await self._handle_unsubscription(user_id, data)
            elif message_type == 'ping':
                # Handle ping messages
                await self.send_to_user(user_id, {'type': 'pong'})

        except Exception as e:
            print(f"Error handling client message: {e}")
            await self.send_to_user(user_id, {
                'type': 'error',
                'data': {'message': str(e)}
            })

    async def _handle_subscription(self, user_id: int, data: dict):
        """Handle client subscription requests"""
        channels = data.get('channels', [])
        symbols = data.get('symbols', [])

        # Store subscription preferences and send initial data
        pass

    async def _handle_unsubscription(self, user_id: int, data: dict):
        """Handle client unsubscription requests"""
        channels = data.get('channels', [])
        symbols = data.get('symbols', [])

        # Remove subscription preferences
        pass

# Initialize WebSocket manager
ws_manager = WebSocketManager()

def init_websocket(app):
    """Initialize WebSocket functionality"""
    sock = Sock(app)
    
    @sock.route('/ws')
    def ws_handler(ws):
        """Handle WebSocket connections"""
        if not getattr(ws, 'user_id', None):
            ws.close()
            return
            
        try:
            # Register client
            ws_manager.register_client(ws.user_id, ws)
            
            while True:
                # Receive and handle messages
                message = ws.receive()
                if message:
                    data = json.loads(message)
                    asyncio.create_task(
                        ws_manager.handle_client_message(ws.user_id, data)
                    )
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            ws_manager.unregister_client(ws.user_id, ws)
    
    return sock