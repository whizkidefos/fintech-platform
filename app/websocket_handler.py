from flask_sock import Sock
import json
from datetime import datetime
from typing import Dict, Set, Optional
import asyncio
import threading
from queue import Queue
from app.utils.market_data import MarketDataFetcher
from app.models.portfolio import Portfolio
from app.models.position import Position
from decimal import Decimal

class WebSocketManager:
    def __init__(self):
        self.clients: Dict[int, Set] = {}  # user_id -> set of websocket connections
        self.subscriptions: Dict[str, Set] = {}  # symbol -> set of user_ids
        self.market_data = MarketDataFetcher()
        self.message_queue = Queue()
        self.running = False
        self.update_thread = None

    def start(self):
        """Start the WebSocket manager"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._process_queue)
            self.update_thread.daemon = True
            self.update_thread.start()

    def stop(self):
        """Stop the WebSocket manager"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()

    def register_client(self, user_id: int, ws):
        """Register a new websocket connection for a user"""
        if user_id not in self.clients:
            self.clients[user_id] = set()
        self.clients[user_id].add(ws)

        if len(self.clients) == 1:
            self.start()

    def unregister_client(self, user_id: int, ws):
        """Unregister a websocket connection"""
        if user_id in self.clients:
            self.clients[user_id].discard(ws)
            if not self.clients[user_id]:
                del self.clients[user_id]
                # Remove user's subscriptions
                for symbol in list(self.subscriptions.keys()):
                    if user_id in self.subscriptions[symbol]:
                        self.subscriptions[symbol].discard(user_id)
                        if not self.subscriptions[symbol]:
                            del self.subscriptions[symbol]

        if not self.clients:
            self.stop()

    def subscribe(self, user_id: int, symbol: str):
        """Subscribe a user to updates for a symbol"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = set()
        self.subscriptions[symbol].add(user_id)

    def unsubscribe(self, user_id: int, symbol: str):
        """Unsubscribe a user from updates for a symbol"""
        if symbol in self.subscriptions:
            self.subscriptions[symbol].discard(user_id)
            if not self.subscriptions[symbol]:
                del self.subscriptions[symbol]

    async def broadcast_market_data(self, symbol: str, data: dict):
        """Broadcast market data to all subscribed users"""
        if symbol in self.subscriptions:
            message = {
                'type': 'market_data',
                'symbol': symbol,
                'data': data
            }
            for user_id in self.subscriptions[symbol]:
                await self._send_to_user(user_id, message)

    async def broadcast_portfolio_update(self, user_id: int, portfolio: Portfolio):
        """Broadcast portfolio updates to a specific user"""
        message = {
            'type': 'portfolio_update',
            'data': {
                'total_value': float(portfolio.total_value),
                'cash_balance': float(portfolio.cash_balance),
                'daily_pnl': float(portfolio.daily_pnl),
                'daily_change': float(portfolio.daily_change)
            }
        }
        await self._send_to_user(user_id, message)

    async def broadcast_position_update(self, position: Position):
        """Broadcast position updates to the position owner"""
        current_price = self.market_data.get_ticker(position.symbol)['last']
        message = {
            'type': 'position_update',
            'data': {
                'id': position.id,
                'symbol': position.symbol,
                'side': position.side,
                'size': float(position.size),
                'entry_price': float(position.entry_price),
                'current_price': float(current_price),
                'pnl': float(position.calculate_pnl(Decimal(str(current_price)))),
                'pnl_percent': float(position.calculate_pnl_percent(Decimal(str(current_price))))
            }
        }
        await self._send_to_user(position.portfolio.user_id, message)

    async def _send_to_user(self, user_id: int, message: dict):
        """Send a message to all connections of a specific user"""
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

    def _process_queue(self):
        """Process messages in the queue"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                asyncio.run(self._process_message(message))
            except Exception:
                continue

    async def _process_message(self, message: dict):
        """Process a single message"""
        message_type = message.get('type')
        
        if message_type == 'market_data':
            await self.broadcast_market_data(message['symbol'], message['data'])
        elif message_type == 'portfolio_update':
            await self.broadcast_portfolio_update(message['user_id'], message['portfolio'])
        elif message_type == 'position_update':
            await self.broadcast_position_update(message['position'])

# Initialize WebSocket manager
ws_manager = WebSocketManager()

def init_websocket(app):
    """Initialize WebSocket functionality"""
    sock = Sock(app)

    @sock.route('/ws/market')
    def market_socket(ws):
        """Handle market data WebSocket connections"""
        if not hasattr(ws, 'user_id'):
            ws.close()
            return

        ws_manager.register_client(ws.user_id, ws)
        
        try:
            while True:
                data = json.loads(ws.receive())
                action = data.get('action')
                symbol = data.get('symbol')

                if action == 'subscribe' and symbol:
                    ws_manager.subscribe(ws.user_id, symbol)
                elif action == 'unsubscribe' and symbol:
                    ws_manager.unsubscribe(ws.user_id, symbol)
        except Exception:
            pass
        finally:
            ws_manager.unregister_client(ws.user_id, ws)