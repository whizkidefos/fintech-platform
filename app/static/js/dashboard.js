class DashboardManager {
    constructor() {
        this.charts = {};
        this.intervals = {};
        this.init();
    }

    init() {
        this.initPortfolioChart();
        this.initWatchlist();
        this.initSignals();
        this.setupWebSocket();
        this.setupThemeToggle();
    }

    async initPortfolioChart() {
        const response = await fetch('/api/portfolio');
        const data = await response.json();
        
        if (data.status === 'success') {
            const portfolioData = this.processPortfolioData(data.data);
            this.renderPortfolioChart(portfolioData);
        }
    }

    processPortfolioData(portfolioData) {
        const processed = {
            labels: [],
            values: [],
            assets: {}
        };
        
        portfolioData.forEach(portfolio => {
            portfolio.assets.forEach(asset => {
                if (!processed.assets[asset.symbol]) {
                    processed.assets[asset.symbol] = [];
                }
                processed.assets[asset.symbol].push({
                    value: asset.value,
                    quantity: asset.quantity,
                    price: asset.current_price
                });
            });
        });

        return processed;
    }

    renderPortfolioChart(data) {
        const ctx = document.getElementById('portfolioChart').getContext('2d');
        
        if (this.charts.portfolio) {
            this.charts.portfolio.destroy();
        }

        this.charts.portfolio = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Portfolio Value',
                    data: data.values,
                    borderColor: getComputedStyle(document.documentElement)
                        .getPropertyValue('--primary-color').trim(),
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `$${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return `$${value.toLocaleString()}`;
                            }
                        }
                    }
                }
            }
        });
    }

    async initWatchlist() {
        const updateWatchlist = async () => {
            const response = await fetch('/api/watchlist');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateWatchlistUI(data.data);
            }
        };

        await updateWatchlist();
        this.intervals.watchlist = setInterval(updateWatchlist, 30000);
    }

    updateWatchlistUI(data) {
        const container = document.querySelector('.watchlist-items');
        container.innerHTML = '';

        data.forEach(asset => {
            const priceChange = ((asset.current_price - asset.previous_price) / 
                               asset.previous_price * 100).toFixed(2);
            const changeClass = priceChange >= 0 ? 'positive' : 'negative';

            container.innerHTML += `
                <div class="watchlist-item">
                    <div class="asset-info">
                        <span class="asset-symbol">${asset.symbol}</span>
                        <span class="asset-name">${asset.name}</span>
                    </div>
                    <div class="asset-price">
                        <span class="price">$${asset.current_price.toLocaleString()}</span>
                        <span class="price-change ${changeClass}">
                            ${priceChange}%
                        </span>
                    </div>
                </div>
            `;
        });
    }

    async initSignals() {
        const updateSignals = async () => {
            const assets = document.querySelectorAll('.watchlist-item .asset-symbol');
            const symbols = Array.from(assets).map(el => el.textContent);

            for (const symbol of symbols) {
                const response = await fetch(`/api/signals/${symbol}`);
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.updateSignalsUI(data.data);
                }
            }
        };

        await updateSignals();
        this.intervals.signals = setInterval(updateSignals, 60000);
    }

    updateSignalsUI(signals) {
        const container = document.querySelector('.signals-list');
        container.innerHTML = '';

        signals.forEach(signal => {
            container.innerHTML += `
                <div class="signal-item ${signal.type}">
                    <div class="signal-icon">
                        <i class="fas fa-${signal.type === 'buy' ? 'arrow-up' : 'arrow-down'}"></i>
                    </div>
                    <div class="signal-details">
                        <span class="signal-asset">${signal.symbol}</span>
                        <span class="signal-type">${signal.type.toUpperCase()}</span>
                        <span class="signal-strength">
                            Strength: ${(signal.strength * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="signal-time">
                        ${new Date(signal.timestamp).toLocaleTimeString()}
                    </div>
                </div>
            `;
        });
    }

    setupWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'price_update':
                    this.handlePriceUpdate(data.data);
                    break;
                case 'signal':
                    this.handleNewSignal(data.data);
                    break;
                case 'portfolio_update':
                    this.handlePortfolioUpdate(data.data);
                    break;
            }
        };

        ws.onclose = () => {
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.setupWebSocket(), 5000);
        };
    }

    handlePriceUpdate(data) {
        const priceElements = document.querySelectorAll(
            `.watchlist-item[data-symbol="${data.symbol}"] .price`
        );
        
        priceElements.forEach(el => {
            const oldPrice = parseFloat(el.textContent.replace('$', ''));
            el.textContent = `$${data.price.toLocaleString()}`;
            
            // Add price flash effect
            el.classList.add(data.price > oldPrice ? 'flash-green' : 'flash-red');
            setTimeout(() => {
                el.classList.remove('flash-green', 'flash-red');
            }, 1000);
        });
    }

    handleNewSignal(signal) {
        const container = document.querySelector('.signals-list');
        const signalElement = document.createElement('div');
        signalElement.className = `signal-item ${signal.type}`;
        signalElement.innerHTML = `
            <div class="signal-icon">
                <i class="fas fa-${signal.type === 'buy' ? 'arrow-up' : 'arrow-down'}"></i>
            </div>
            <div class="signal-details">
                <span class="signal-asset">${signal.symbol}</span>
                <span class="signal-type">${signal.type.toUpperCase()}</span>
                <span class="signal-strength">
                    Strength: ${(signal.strength * 100).toFixed(1)}%
                </span>
            </div>
            <div class="signal-time">
                ${new Date(signal.timestamp).toLocaleTimeString()}
            </div>
        `;

        container.insertBefore(signalElement, container.firstChild);
        
        // Remove oldest signal if more than 5 are displayed
        if (container.children.length > 5) {
            container.removeChild(container.lastChild);
        }
    }

    handlePortfolioUpdate(data) {
        this.renderPortfolioChart(this.processPortfolioData(data));
    }

    setupThemeToggle() {
        const toggle = document.getElementById('theme-toggle');
        toggle.addEventListener('click', () => {
            const body = document.body;
            const currentTheme = body.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            
            body.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update chart themes
            this.updateChartThemes(newTheme);
        });
    }

    updateChartThemes(theme) {
        const textColor = theme === 'dark' ? '#ffffff' : '#333333';
        const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

        Object.values(this.charts).forEach(chart => {
            chart.options.scales.x.grid.color = gridColor;
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.y.ticks.color = textColor;
            chart.update();
        });
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    new DashboardManager();
});