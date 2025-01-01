class AnalyticsDashboard {
    constructor() {
        this.charts = new Map();
        this.currentTimeframe = '1M';
        this.initializeCharts();
        this.initializeEventListeners();
    }

    initializeCharts() {
        // Equity Curve Chart
        this.charts.set('equity', this.createEquityCurveChart());
        
        // Risk Distribution Chart
        this.charts.set('risk', this.createRiskDistributionChart());
        
        // Performance Charts
        this.charts.set('hourly', this.createPerformanceChart('hourlyPerformanceChart', 'Hourly Performance'));
        this.charts.set('daily', this.createPerformanceChart('dailyPerformanceChart', 'Daily Performance'));
        this.charts.set('symbol', this.createPerformanceChart('symbolPerformanceChart', 'Symbol Performance'));
        this.charts.set('strategy', this.createPerformanceChart('strategyPerformanceChart', 'Strategy Performance'));
    }

    createEquityCurveChart() {
        const ctx = document.getElementById('equityCurveChart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: (context) => {
                                return `$${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day'
                        }
                    },
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    }

    createRiskDistributionChart() {
        const ctx = document.getElementById('riskDistributionChart').getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Return Distribution',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    borderColor: 'rgb(75, 192, 192)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    createPerformanceChart(canvasId, title) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'PnL',
                    data: [],
                    backgroundColor: (context) => {
                        const value = context.raw;
                        return value >= 0 ? 'rgba(75, 192, 192, 0.5)' : 'rgba(255, 99, 132, 0.5)';
                    },
                    borderColor: (context) => {
                        const value = context.raw;
                        return value >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)';
                    },
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    async updateCharts() {
        try {
            const response = await fetch(`/api/analytics?timeframe=${this.currentTimeframe}`);
            const data = await response.json();
            
            // Update Equity Curve
            this.updateEquityCurve(data.equity_curve);
            
            // Update Risk Distribution
            this.updateRiskDistribution(data.returns_distribution);
            
            // Update Performance Charts
            this.updatePerformanceCharts(data.performance);
            
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }

    updateEquityCurve(data) {
        const chart = this.charts.get('equity');
        chart.data.labels = data.dates;
        chart.data.datasets[0].data = data.values;
        chart.update();
    }

    updateRiskDistribution(data) {
        const chart = this.charts.get('risk');
        chart.data.labels = data.bins;
        chart.data.datasets[0].data = data.frequencies;
        chart.update();
    }

    updatePerformanceCharts(data) {
        // Update Hourly Performance
        const hourlyChart = this.charts.get('hourly');
        hourlyChart.data.labels = data.hourly.labels;
        hourlyChart.data.datasets[0].data = data.hourly.values;
        hourlyChart.update();
        
        // Update Daily Performance
        const dailyChart = this.charts.get('daily');
        dailyChart.data.labels = data.daily.labels;
        dailyChart.data.datasets[0].data = data.daily.values;
        dailyChart.update();
        
        // Update Symbol Performance
        const symbolChart = this.charts.get('symbol');
        symbolChart.data.labels = data.symbol.labels;
        symbolChart.data.datasets[0].data = data.symbol.values;
        symbolChart.update();
        
        // Update Strategy Performance
        const strategyChart = this.charts.get('strategy');
        strategyChart.data.labels = data.strategy.labels;
        strategyChart.data.datasets[0].data = data.strategy.values;
        strategyChart.update();
    }

    initializeEventListeners() {
        // Timeframe selector
        document.getElementById('timeframeSelector').addEventListener('change', (e) => {
            this.currentTimeframe = e.target.value;
            this.updateCharts();
        });
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
        
        // Position calculator inputs
        const calculatorInputs = ['riskPerTradeInput', 'entryPriceInput', 'stopLossInput'];
        calculatorInputs.forEach(inputId => {
            document.getElementById(inputId).addEventListener('input', () => {
                this.calculatePosition();
            });
        });
    }

    switchTab(tabId) {
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        
        // Update active tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.toggle('active', pane.id === tabId + 'Analysis');
        });
        
        // Trigger resize on charts to fix any layout issues
        this.charts.forEach(chart => chart.resize());
    }

    calculatePosition() {
        const accountBalance = parseFloat(document.getElementById('accountBalanceSpan').textContent.replace('$', ''));
        const riskPercent = parseFloat(document.getElementById('riskPerTradeInput').value) / 100;
        const entryPrice = parseFloat(document.getElementById('entryPriceInput').value);
        const stopLoss = parseFloat(document.getElementById('stopLossInput').value);
        
        if (!isNaN(accountBalance) && !isNaN(riskPercent) && !isNaN(entryPrice) && !isNaN(stopLoss)) {
            const riskAmount = accountBalance * riskPercent;
            const priceDiff = Math.abs(entryPrice - stopLoss);
            const positionSize = priceDiff !== 0 ? riskAmount / priceDiff : 0;
            
            document.getElementById('positionSizeResult').textContent = positionSize.toFixed(8);
            document.getElementById('totalRiskResult').textContent = `$${riskAmount.toFixed(2)}`;
        }
    }

    async exportTradeData() {
        try {
            const response = await fetch(`/api/analytics/export?timeframe=${this.currentTimeframe}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `trade_analysis_${this.currentTimeframe}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } catch (error) {
            console.error('Error exporting data:', error);
        }
    }
}

// Initialize dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.analyticsDashboard = new AnalyticsDashboard();
});
