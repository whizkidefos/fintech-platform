# Fintech Platform

A comprehensive fintech platform built with Flask, featuring real-time market data analysis, automated trading signals, and portfolio management.

## Features

- Real-time market data tracking for stocks, assets, and cryptocurrencies
- Automated trading signals based on technical analysis
- Interactive dashboard with dynamic charts
- Portfolio management and tracking
- Light/Dark theme support
- WebSocket integration for real-time updates
- Secure authentication system
- RESTful API endpoints
- SQLite database (PostgreSQL ready)

## Technology Stack

- **Backend**: Flask, SQLAlchemy, WebSockets
- **Frontend**: HTML5, Sass, JavaScript
- **Database**: SQLite (PostgreSQL ready)
- **Real-time Data**: WebSocket
- **Market Data**: CCXT, Python-Binance
- **Styling**: Sass with theme support

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fintech-platform
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Node.js dependencies:
   ```bash
   npm install
   ```

5. Create .env file:
   ```
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=sqlite:///fintech.db
   BINANCE_API_KEY=your-binance-api-key
   BINANCE_SECRET_KEY=your-binance-secret-key
   ```

6. Initialize the database:
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

## Development

Start the development server with:
```bash
npm run dev
```

This command will:
- Start the Flask development server
- Watch for Sass changes and compile to CSS
- Enable hot reloading

## Project Structure

```
fintech-platform/
├── app/
│   ├── models/          # Database models
│   ├── routes/          # Route handlers
│   ├── static/          # Static files (JS, Sass, images)
│   ├── templates/       # HTML templates
│   └── utils/           # Utility functions
├── migrations/          # Database migrations
├── node_modules/        # Node.js dependencies
├── .env                 # Environment variables
├── config.py           # Application configuration
├── requirements.txt    # Python dependencies
└── run.py             # Application entry point
```

## API Documentation

The platform provides several API endpoints:

- `/api/market-data/<symbol>` - Get market data for a specific symbol
- `/api/signals/<symbol>` - Get trading signals for a specific symbol
- `/api/portfolio` - Get user portfolio data
- `/api/watchlist` - Get watchlist data
- `/api/execute-trade` - Execute a trade

Detailed API documentation is available in the `/docs` directory.

## WebSocket Events

The platform uses WebSocket for real-time updates:

- `price_update` - Real-time price updates
- `signal` - New trading signals
- `portfolio_update` - Portfolio value updates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency exchange support
- [Chart.js](https://www.chartjs.org/) for interactive charts
- [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/) for database management