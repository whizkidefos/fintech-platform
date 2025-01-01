// Connect to WebSocket
const ws = new WebSocket(`ws://${window.location.host}/ws`);

// Subscribe to symbols
ws.send(JSON.stringify({
    type: 'subscribe',
    symbols: ['BTC/USDT', 'ETH/USDT']
}));

// Request specific updates
ws.send(JSON.stringify({
    type: 'get_price',
    symbol: 'BTC/USDT'
}));

// Handle incoming messages
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch(data.type) {
        case 'price_update':
            handlePriceUpdate(data.data);
            break;
        case 'signals_update':
            handleSignalsUpdate(data.data);
            break;
    }
};