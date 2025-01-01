class ChartManager {
    // ... (previous methods remain the same)

    handleMarketUpdate(symbol, data) {
        const chart = this.charts.get(symbol);
        if (!chart) return;

        const lastPoint = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];
        const newTimestamp = new Date(data.timestamp);

        // Update last candle if within same timeframe, otherwise add new candle
        if (lastPoint && this.isSameTimeframe(lastPoint.x, newTimestamp)) {
            lastPoint.high = Math.max(lastPoint.high, data.price);
            lastPoint.low = Math.min(lastPoint.low, data.price);
            lastPoint.close = data.price;
            lastPoint.volume += data.volume;
        } else {
            chart.data.datasets[0].data.push({
                x: newTimestamp,
                open: data.price,
                high: data.price,
                low: data.price,
                close: data.price,
                volume: data.volume
            });

            // Remove oldest candle if we have too many
            if (chart.data.datasets[0].data.length > this.getMaxCandles()) {
                chart.data.datasets[0].data.shift();
            }
        }

        chart.update('quiet');
    }

    isSameTimeframe(timestamp1, timestamp2) {
        const interval = this.timeframes[this.currentTimeframe];
        return Math.floor(timestamp1 / interval) === Math.floor(timestamp2 / interval);
    }

    getMaxCandles() {
        switch (this.currentTimeframe) {
            case '1m': return 60;
            case '5m': return 72;
            case '15m': return 96;
            case '1h': return 100;
            case '4h': return 120;
            case '1d': return 90;
            default: return 100;
        }
    }

    // Technical indicators
    addIndicator(symbol, type, params = {}) {
        const chart = this.charts.get(symbol);
        if (!chart) return;

        switch (type) {
            case 'sma':
                this.addSMA(chart, params.period || 20);
                break;
            case 'ema':
                this.addEMA(chart, params.period || 20);
                break;
            case 'bollinger':
                this.addBollingerBands(chart, params.period || 20, params.stdDev || 2);
                break;
            case 'volume':
                this.addVolumeIndicator(chart);
                break;
        }
    }

    addSMA(chart, period) {
        const prices = chart.data.datasets[0].data.map(d => d.close);
        const sma = this.calculateSMA(prices, period);
        
        chart.data.datasets.push({
            label: `SMA ${period}`,
            data: sma.map((value, index) => ({
                x: chart.data.datasets[0].data[index].x,
                y: value
            })),
            type: 'line',
            borderColor: 'rgba(255, 99, 132, 1)',
            tension: 0.4,
            fill: false
        });

        chart.update();
    }

    addEMA(chart, period) {
        const prices = chart.data.datasets[0].data.map(d => d.close);
        const ema = this.calculateEMA(prices, period);
        
        chart.data.datasets.push({
            label: `EMA ${period}`,
            data: ema.map((value, index) => ({
                x: chart.data.datasets[0].data[index].x,
                y: value
            })),
            type: 'line',
            borderColor: 'rgba(54, 162, 235, 1)',
            tension: 0.4,
            fill: false
        });

        chart.update();
    }

    addBollingerBands(chart, period, stdDev) {
        const prices = chart.data.datasets[0].data.map(d => d.close);
        const { upper, middle, lower } = this.calculateBollingerBands(prices, period, stdDev);
        
        // Middle Band (SMA)
        chart.data.datasets.push({
            label: `BB Middle (${period})`,
            data: middle.map((value, index) => ({
                x: chart.data.datasets[0].data[index].x,
                y: value
            })),
            type: 'line',
            borderColor: 'rgba(75, 192, 192, 1)',
            tension: 0.4,
            fill: false
        });

        // Upper Band
        chart.data.datasets.push({
            label: `BB Upper (${period}, ${stdDev}σ)`,
            data: upper.map((value, index) => ({
                x: chart.data.datasets[0].data[index].x,
                y: value
            })),
            type: 'line',
            borderColor: 'rgba(75, 192, 192, 0.5)',
            tension: 0.4,
            fill: false
        });

        // Lower Band
        chart.data.datasets.push({
            label: `BB Lower (${period}, ${stdDev}σ)`,
            data: lower.map((value, index) => ({
                x: chart.data.datasets[0].data[index].x,
                y: value
            })),
            type: 'line',
            borderColor: 'rgba(75, 192, 192, 0.5)',
            tension: 0.4,
            fill: false
        });

        chart.update();
    }

    addVolumeIndicator(chart) {
        const volumeData = chart.data.datasets[0].data.map(d => ({
            x: d.x,
            y: d.volume,
            color: d.close >= d.open ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)'
        }));

        chart.data.datasets.push({
            label: 'Volume',
            data: volumeData,
            type: 'bar',
            yAxisID: 'volume',
            backgroundColor: volumeData.map(d => d.color)
        });

        // Add volume scale
        chart.options.scales.volume = {
            position: 'left',
            grid: {
                drawOnChartArea: false
            },
            ticks: {
                callback: function(value) {
                    return formatNumber(value);
                }
            }
        };

        chart.update();
    }

    // Technical indicator calculations
    calculateSMA(data, period) {
        const sma = [];
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                sma.push(null);
                continue;
            }
            
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            sma.push(sum / period);
        }
        return sma;
    }

    calculateEMA(data, period) {
        const ema = [];
        const multiplier = 2 / (period + 1);

        // First EMA is SMA
        let prevEMA = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
        ema.push(prevEMA);

        for (let i = period; i < data.length; i++) {
            const currentEMA = (data[i] - prevEMA) * multiplier + prevEMA;
            ema.push(currentEMA);
            prevEMA = currentEMA;
        }

        return Array(period - 1).fill(null).concat(ema);
    }

    calculateBollingerBands(data, period, stdDev) {
        const middle = this.calculateSMA(data, period);
        const upper = [];
        const lower = [];

        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                upper.push(null);
                lower.push(null);
                continue;
            }

            const slice = data.slice(i - period + 1, i + 1);
            const std = this.calculateStandardDeviation(slice);
            
            upper.push(middle[i] + (std * stdDev));
            lower.push(middle[i] - (std * stdDev));
        }

        return { upper, middle, lower };
    }

    calculateStandardDeviation(data) {
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const squareDiffs = data.map(value => Math.pow(value - mean, 2));
        const variance = squareDiffs.reduce((a, b) => a + b, 0) / data.length;
        return Math.sqrt(variance);
    }
}

// Initialize chart manager when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chartManager = new ChartManager();
    window.chartManager.initializeCharts();

    // Set up timeframe selector
    const timeframeSelector = document.getElementById('timeframeSelector');
    if (timeframeSelector) {
        timeframeSelector.addEventListener('change', (e) => {
            window.chartManager.updateTimeframe(e.target.value);
        });
    }

    // Set up indicator controls
    document.querySelectorAll('[data-indicator]').forEach(button => {
        button.addEventListener('click', () => {
            const symbol = button.dataset.symbol;
            const indicator = button.dataset.indicator;
            const params = JSON.parse(button.dataset.params || '{}');
            window.chartManager.addIndicator(symbol, indicator, params);
        });
    });
});