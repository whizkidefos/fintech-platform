from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config
import atexit

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()

def cleanup_resources():
    """Cleanup function to be called on application exit"""
    try:
        if db.session.registry().has():
            db.session.remove()
    except Exception as e:
        print(f"Cleanup error: {e}")

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    # Register cleanup function
    atexit.register(cleanup_resources)

    # Import models to ensure they are registered with SQLAlchemy
    with app.app_context():
        from app.models import User, Portfolio, Position, Transaction, TradingSignal, Asset

        # Register blueprints
        from app.routes import auth, dashboard, api
        app.register_blueprint(auth.bp)
        app.register_blueprint(dashboard.bp)
        app.register_blueprint(api.bp)

        # Create database tables
        db.create_all()

    return app