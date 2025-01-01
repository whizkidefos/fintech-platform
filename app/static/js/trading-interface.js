class TradingInterface {
    constructor() {
        this.positions = new Map();
        this.orders = new Map();
        this.activeSymbol = null;
        this.orderTypes = ['MARKET', 'LIMIT', 'STOP', 'OCO'];
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Quick trade panel events
        document.getElementById('quickTradeBtn')?.addEventListener('click', () => this.toggleQuickTradePanel());
        document.getElementById('submitTradeBtn')?.addEventListener('click', () => this.submitQuickTrade());

        // Order form events
        document.getElementById('orderTypeSelect')?.addEventListener('change', (e) => this.handleOrderTypeChange(e));
        document.getElementById('positionSizeInput')?.addEventListener('input', (e) => this.updatePositionSizing(e));
        
        // Risk calculator events
        document.getElementById('riskPercentInput')?.addEventListener('input', (e) => this.calculatePositionSize(e));
        document.getElementById('stopLossInput')?.addEventListener('input', (e) => this.calculatePositionSize(e));
    }

    toggleQuickTradePanel() {
        const panel = document.getElementById('quickTradePanel');
        if (panel) {
            panel.classList.toggle('show');
            if (panel.classList.contains('show')) {
                this.updateQuickTradePrices();
            }
        }
    }

    async submitQuickTrade() {
        const symbol = document.getElementById('symbolInput').value;
        const side = document.getElementById('tradeSideSelect').value;
        const size = document.getElementById('positionSizeInput').value;
        const type = document.getElementById('orderTypeSelect').value;
        
        const orderData = {
            symbol,
            side,
            size: parseFloat(size),
            type,
            timestamp: new Date().toISOString()
        };

        if (type === 'LIMIT' || type === 'STOP') {
            orderData.price = parseFloat(document.getElementById('limitPriceInput').value);
        }

        if (document.getElementById('stopLossEnabled').checked) {
            orderData.stopLoss = parseFloat(document.getElementById('stopLossInput').value);
        }

        if (document.getElementById('takeProfitEnabled').checked) {
            orderData.takeProfit = parseFloat(document.getElementById('takeProfitInput').value);
        }

        try {
            const response = await this.submitOrder(orderData);
            if (response.success) {
                this.showNotification('Order submitted successfully', 'success');
                this.toggleQuickTradePanel();
            } else {
                this.showNotification('Failed to submit order: ' + response.message, 'error');
            }
        } catch (error) {
            this.showNotification('Error submitting order: ' + error.message, 'error');
        }
    }

    async submitOrder(orderData) {
        try {
            const response = await fetch('/api/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(orderData)
            });

            return await response.json();
        } catch (error) {
            console.error('Error submitting order:', error);
            throw error;
        }
    }

    handleOrderTypeChange(event) {
        const orderType = event.target.value;
        const limitPriceContainer = document.getElementById('limitPriceContainer');
        const ocoContainer = document.getElementById('ocoContainer');

        if (limitPriceContainer) {
            limitPriceContainer.style.display = 
                ['LIMIT', 'STOP', 'OCO'].includes(orderType) ? 'block' : 'none';
        }

        if (ocoContainer) {
            ocoContainer.style.display = orderType === 'OCO' ? 'block' : 'none';
        }
    }

    updatePositionSizing(event) {
        const size = parseFloat(event.target.value);
        const price = parseFloat(document.getElementById('currentPriceSpan').textContent);
        
        if (!isNaN(size) && !isNaN(price)) {
            const positionValue = size * price;
            document.getElementById('positionValueSpan').textContent = 
                positionValue.toFixed(2);
        }
    }

    calculatePositionSize(event) {
        const accountBalance = parseFloat(document.getElementById('accountBalanceSpan').textContent);
        const riskPercent = parseFloat(document.getElementById('riskPercentInput').value);
        const entryPrice = parseFloat(document.getElementById('currentPriceSpan').textContent);
        const stopLoss = parseFloat(document.getElementById('stopLossInput').value);

        if (!isNaN(accountBalance) && !isNaN(riskPercent) && 
            !isNaN(entryPrice) && !isNaN(stopLoss)) {
            const riskAmount = accountBalance * (riskPercent / 100);
            const priceDiff = Math.abs(entryPrice - stopLoss);
            const positionSize = riskAmount / priceDiff;

            document.getElementById('positionSizeInput').value = positionSize.toFixed(4);
            this.updatePositionSizing({ target: { value: positionSize } });
        }
    }

    async updateQuickTradePrices() {
        const symbol = document.getElementById('symbolInput').value;
        if (!symbol) return;

        try {
            const response = await fetch(`/api/market-data/${symbol}/ticker`);
            const data = await response.json();

            document.getElementById('currentPriceSpan').textContent = data.last;
            document.getElementById('bidPriceSpan').textContent = data.bid;
            document.getElementById('askPriceSpan').textContent = data.ask;
        } catch (error) {
            console.error('Error updating prices:', error);
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 5000);
    }
}

// Initialize trading interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.tradingInterface = new TradingInterface();
});
