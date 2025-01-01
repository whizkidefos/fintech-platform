import json
from flask_sock import Sock
from datetime import datetime
from app.utils.market_data import MarketDataFetcher
from app.utils.trading_signals import SignalGenerator

class WebSocketManager:
    def __init__(self):
        self.clients = set()
        self.market_data = MarketDataFetcher()
        self.running = False

    def handle_client(self, ws):
        """Handle individual WebSocket client connections"""
        self.clients.add(ws)
        try:
            while True:
                try:
                    # Handle incoming messages
                    message = ws.receive()
                    if message:
                        data = json.loads(message)
                        self.process_message(ws, data)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.clients.remove(ws)

    def process_message(self, ws, data):
        """Process incoming WebSocket messages"""
        try:
            message_type = data.get('type')
            if message_type == 'subscribe':
                self.handle_subscription(ws, data)
            elif message_type == 'get_price':
                self.handle_price_request(ws, data)
            elif message_type == 'get_signals':
                self.handle_signal_request(ws, data)
        except Exception as e:
            print(f"Error processing message: {e}")

    def handle_subscription(self, ws, data):
        """Handle subscription requests"""
        symbols = data.get('symbols', [])
        if symbols:
            # Send initial data for subscribed symbols
            for symbol in symbols:
                self.send_price_update(ws, symbol)
                self.send_signals_update(ws, symbol)

    def handle_price_request(self, ws, data):
        """Handle price update requests"""
        symbol = data.get('symbol')
        if symbol:
            self.send_price_update(ws, symbol)

    def handle_signal_request(self, ws, data):
        """Handle signal requests"""
        symbol = data.get('symbol')
        if symbol:
            self.send_signals_update(ws, symbol)

    def send_price_update(self, ws, symbol):
        """Send price updates to client"""
        try:
            ticker = self.market_data.fetch_ticker(symbol)
            if ticker:
                ws.send(json.dumps({
                    'type': 'price_update',
                    'data': {
                        'symbol': symbol,
                        'price': ticker['last_price'],
                        'timestamp': datetime.now().isoformat()
                    }
                }))
        except Exception as e:
            print(f"Error sending price update: {e}")

    def send_signals_update(self, ws, symbol):
        """Send trading signals to client"""
        try:
            data = self.market_data.fetch_ohlcv(symbol, '1m', limit=100)
            if data is not None:
                signal_generator = SignalGenerator(data)
                signals = signal_generator.generate_signals()
                
                if signals:
                    ws.send(json.dumps({
                        'type': 'signals_update',
                        'data': {
                            'symbol': symbol,
                            'signals': signals,
                            'timestamp': datetime.now().isoformat()
                        }
                    }))
        except Exception as e:
            print(f"Error sending signals update: {e}")

    def broadcast(self, message):
        """Broadcast message to all connected clients"""
        for client in self.clients.copy():  # Use copy to avoid modification during iteration
            try:
                client.send(json.dumps(message))
            except Exception:
                self.clients.discard(client)

# Create singleton instance
websocket_manager = WebSocketManager()

def init_websocket(app):
    """Initialize WebSocket functionality"""
    sock = Sock(app)
    
    @sock.route('/ws')
    def ws_handler(ws):
        """Handle WebSocket connections"""
        websocket_manager.handle_client(ws)
    
    return sock