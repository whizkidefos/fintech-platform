class WebSocketClient {
    constructor() {
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.handlers = new Map();
        this.subscriptions = new Set();
        this.connect();
    }

    // ... (previous methods remain the same)

    // Utility functions
    function formatCurrency(value, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }

    function formatTimeAgo(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000); // difference in seconds

        if (diff < 60) {
            return `${diff}s ago`;
        } else if (diff < 3600) {
            const minutes = Math.floor(diff / 60);
            return `${minutes}m ago`;
        } else if (diff < 86400) {
            const hours = Math.floor(diff / 3600);
            return `${hours}h ago`;
        } else {
            const days = Math.floor(diff / 86400);
            return `${days}d ago`;
        }
    }

    function formatNumber(value) {
        return new Intl.NumberFormat('en-US', {
            maximumFractionDigits: 2,
            notation: 'compact',
            compactDisplay: 'short'
        }).format(value);
    }

    // Portfolio Chart Updates
    function updatePortfolioChart(data) {
        if (!window.portfolioChart) return;

        const chart = window.portfolioChart;
        const dataset = chart.data.datasets[0];

        // Add new data point
        chart.data.labels.push(new Date(data.timestamp));
        dataset.data.push(data.total_value);

        // Remove oldest point if we have more than 100 points
        if (chart.data.labels.length > 100) {
            chart.data.labels.shift();
            dataset.data.shift();
        }

        // Update chart
        chart.update('quiet');
    }

    // Signal Execution
    async function executeSignal(signalId) {
        try {
            const response = await fetch('/api/signals/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ signal_id: signalId })
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                showNotification('success', 'Signal executed successfully');
                // Update UI
                const signalElement = document.querySelector(`[data-signal-id="${signalId}"]`);
                if (signalElement) {
                    signalElement.classList.add('executed');
                }
            } else {
                showNotification('error', result.message);
            }
        } catch (error) {
            showNotification('error', 'Failed to execute signal');
            console.error('Error executing signal:', error);
        }
    }

    // Notification System
    function showNotification(type, message) {
        const container = document.getElementById('notification-container');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
            <span>${message}</span>
        `;

        container.appendChild(notification);

        // Remove notification after 5 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => {
                container.removeChild(notification);
            }, 300);
        }, 5000);
    }

    // Asset Search
    function setupAssetSearch() {
        const searchInput = document.querySelector('.market-overview input');
        if (!searchInput) return;

        let debounceTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(debounceTimeout);
            debounceTimeout = setTimeout(() => {
                const searchTerm = e.target.value.toLowerCase();
                filterAssets(searchTerm);
            }, 300);
        });
    }

    function filterAssets(searchTerm) {
        const assetItems = document.querySelectorAll('.market-item');
        assetItems.forEach(item => {
            const symbol = item.querySelector('.asset-symbol').textContent.toLowerCase();
            const name = item.querySelector('.asset-name').textContent.toLowerCase();
            
            if (symbol.includes(searchTerm) || name.includes(searchTerm)) {
                item.style.display = '';
            } else {
                item.style.display = 'none';
            }
        });
    }

    // Watchlist Management
    async function toggleWatchlist(symbol) {
        try {
            const response = await fetch('/api/watchlist/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol })
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                const button = document.querySelector(`[data-watchlist-symbol="${symbol}"]`);
                if (button) {
                    button.classList.toggle('active');
                    button.querySelector('i').classList.toggle('fas');
                    button.querySelector('i').classList.toggle('far');
                }
            }
        } catch (error) {
            console.error('Error toggling watchlist:', error);
        }
    }

    // Position Management
    function openNewPositionModal(symbol = null) {
        const modal = document.getElementById('newPositionModal');
        if (!modal) return;

        // Populate form if symbol is provided
        if (symbol) {
            const symbolSelect = modal.querySelector('#positionSymbol');
            if (symbolSelect) {
                symbolSelect.value = symbol;
                updatePositionForm(symbol);
            }
        }

        modal.classList.add('show');
    }

    async function updatePositionForm(symbol) {
        try {
            const response = await fetch(`/api/market/quote/${symbol}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                // Update price fields
                document.getElementById('currentPrice').textContent = 
                    formatCurrency(data.data.price);
                document.getElementById('positionPrice').value = data.data.price;
            }
        } catch (error) {
            console.error('Error updating position form:', error);
        }
    }
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize WebSocket
    window.wsClient = new WebSocketClient();

    // Setup UI components
    setupAssetSearch();

    // Initialize tooltips
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        new Tooltip(tooltip);
    });
});