class ChartManager {
    constructor() {
        this.charts = new Map();
        this.indicators = new Map();
        this.drawings = new Map();
        this.timeframes = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000
        };
        this.currentTimeframe = '15m';
        this.defaultIndicators = ['MA', 'Volume'];
    }

    async initializeChart(containerId, symbol, options = {}) {
        const canvas = document.getElementById(containerId);
        if (!canvas) return null;

        const chartConfig = {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: symbol,
                    data: []
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'DD HH:mm',
                                day: 'MMM DD'
                            }
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        position: 'right',
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: this._formatTooltipLabel.bind(this)
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x'
                        },
                        zoom: {
                            wheel: {
                                enabled: true
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x'
                        }
                    }
                },
                animation: false
            }
        };

        const chart = new Chart(canvas, chartConfig);
        this.charts.set(symbol, chart);
        
        // Initialize default indicators
        await this._initializeDefaultIndicators(symbol);
        
        return chart;
    }

    async addIndicator(symbol, indicatorType, params = {}) {
        const chart = this.charts.get(symbol);
        if (!chart) return;

        const indicator = await this._createIndicator(indicatorType, params);
        if (!indicator) return;

        const indicatorKey = `${symbol}-${indicatorType}-${Date.now()}`;
        this.indicators.set(indicatorKey, indicator);

        // Add indicator dataset to chart
        chart.data.datasets.push(indicator.dataset);
        chart.update('quiet');

        return indicatorKey;
    }

    removeIndicator(indicatorKey) {
        const [symbol] = indicatorKey.split('-');
        const chart = this.charts.get(symbol);
        const indicator = this.indicators.get(indicatorKey);

        if (chart && indicator) {
            const datasetIndex = chart.data.datasets.indexOf(indicator.dataset);
            if (datasetIndex > -1) {
                chart.data.datasets.splice(datasetIndex, 1);
                chart.update('quiet');
            }
            this.indicators.delete(indicatorKey);
        }
    }

    addDrawing(symbol, type, points) {
        const chart = this.charts.get(symbol);
        if (!chart) return;

        const drawingKey = `${symbol}-${type}-${Date.now()}`;
        const drawing = this._createDrawing(type, points);
        
        if (drawing) {
            this.drawings.set(drawingKey, drawing);
            chart.update('quiet');
        }

        return drawingKey;
    }

    removeDrawing(drawingKey) {
        const [symbol] = drawingKey.split('-');
        const chart = this.charts.get(symbol);
        
        if (chart && this.drawings.has(drawingKey)) {
            this.drawings.delete(drawingKey);
            chart.update('quiet');
        }
    }

    async _initializeDefaultIndicators(symbol) {
        for (const indicator of this.defaultIndicators) {
            await this.addIndicator(symbol, indicator);
        }
    }

    async _createIndicator(type, params) {
        // Implementation for different indicator types
        switch (type.toUpperCase()) {
            case 'MA':
                return this._createMovingAverage(params);
            case 'VOLUME':
                return this._createVolumeIndicator(params);
            case 'RSI':
                return this._createRSI(params);
            case 'MACD':
                return this._createMACD(params);
            case 'BBANDS':
                return this._createBollingerBands(params);
            default:
                console.warn(`Unsupported indicator type: ${type}`);
                return null;
        }
    }

    _createDrawing(type, points) {
        // Implementation for different drawing types
        switch (type.toUpperCase()) {
            case 'TRENDLINE':
                return this._createTrendLine(points);
            case 'FIBONACCI':
                return this._createFibonacciRetracement(points);
            case 'RECTANGLE':
                return this._createRectangle(points);
            default:
                console.warn(`Unsupported drawing type: ${type}`);
                return null;
        }
    }

    _formatTooltipLabel(context) {
        const data = context.raw;
        if (!data) return '';

        if (data.o !== undefined) {
            return [
                `Open: ${data.o.toFixed(2)}`,
                `High: ${data.h.toFixed(2)}`,
                `Low: ${data.l.toFixed(2)}`,
                `Close: ${data.c.toFixed(2)}`,
                `Volume: ${data.v.toFixed(2)}`
            ];
        }

        return `${context.dataset.label}: ${context.raw.y.toFixed(2)}`;
    }

    // Additional helper methods for indicators and drawings...
}

// Initialize chart manager when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chartManager = new ChartManager();
});
