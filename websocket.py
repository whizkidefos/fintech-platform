import asyncio
import json
import websockets
from datetime import datetime
from app.utils.market_data import MarketDataFetcher
from app.utils.trading_signals import SignalGenerator

class WebSocketServer:
    def __init__(self):
        self.clients = set()
        self.market_data = MarketDataFetcher()
        self.running = False
        self.tasks = []

    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, message):
        if not self.clients:
            return
        await asyncio.gather(
            *[client.send(json.dumps(message)) for client in self.clients]
        )

    async def price_update_loop(self):
        while self.running:
            try:
                # Fetch current prices for all watched assets
                for symbol in self.get_watched_symbols():
                    ticker = self.market_data.fetch_ticker(symbol)
                    if ticker:
                        await self.broadcast({
                            'type': 'price_update',
                            'data': {
                                'symbol': symbol,
                                'price': ticker['last_price'],
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error in price update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def signal_check_loop(self):
        while self.running:
            try:
                for symbol in self.get_watched_symbols():
                    # Fetch recent OHLCV data
                    data = self.market_data.fetch_ohlcv(symbol, '1m', limit=100)
                    if data is not None:
                        # Generate signals
                        signal_generator = SignalGenerator(data)
                        signals = signal_generator.generate_signals()
                        
                        # Broadcast new signals
                        for signal in signals:
                            await self.broadcast({
                                'type': 'signal',
                                'data': {
                                    'symbol': symbol,
                                    'signal_type': signal['type'],
                                    'strength': signal['strength'],
                                    'indicators': signal['indicators'],
                                    'timestamp': datetime.now().isoformat()
                                }
                            })
                await asyncio.sleep(60)  # Check for signals every minute
            except Exception as e:
                print(f"Error in signal check loop: {e}")
                await asyncio.sleep(5)

    def get_watched_symbols(self):
        # This should be implemented to return the list of symbols being watched
        # Could be fetched from database or configuration
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']  # Example symbols

    def start(self):
        self.running = True
        self.tasks = [
            asyncio.create_task(self.price_update_loop()),
            asyncio.create_task(self.signal_check_loop())
        ]

    def stop(self):
        self.running = False
        for task in self.tasks:
            task.cancel()
        self.tasks = []

websocket_server = WebSocketServer()

# In your Flask application, you'll need to set up the WebSocket handler
# This can be done using Flask-Sock or a similar extension

from flask_sock import Sock

def init_websocket(app):
    sock = Sock(app)

    @sock.route('/ws')
    def ws(ws):
        asyncio.run(websocket_server.register(ws))

    websocket_server.start()
    return sock