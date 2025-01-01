from app import create_app, db
from app.models import User, Portfolio, Position, Transaction, TradingSignal, Asset
import os
import socket
import sys

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return False
        except socket.error:
            return True

def find_available_port(start_port):
    port = start_port
    while is_port_in_use(port) and port < start_port + 100:
        port += 1
    return port

def init_db():
    """Initialize the database with migrations"""
    try:
        # Import Flask-Migrate commands
        from flask_migrate import init, migrate, upgrade
        
        with app.app_context():
            # Check if migrations folder exists
            if not os.path.exists('migrations'):
                # Initialize migrations
                init()
            
            # Generate migration
            migrate()
            
            # Apply migration
            upgrade()
            
            print("Database migration completed successfully!")
    except Exception as e:
        print(f"Error during database migration: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Create Flask application
    app = create_app()
    
    # Initialize database with migrations
    init_db()
    
    # Get port from environment variable or use default
    default_port = 5000
    port = int(os.environ.get('PORT', default_port))
    
    # If the port is in use, find an available one
    if is_port_in_use(port):
        new_port = find_available_port(port + 1)
        if new_port < port + 100:
            print(f"Port {port} is in use. Using port {new_port} instead.")
            port = new_port
        else:
            print(f"Error: Could not find an available port between {port} and {port + 99}")
            sys.exit(1)
    
    try:
        # Run the application
        app.run(host='127.0.0.1', port=port, debug=True)
    except Exception as e:
        print(f"Error starting the server: {e}")
        sys.exit(1)