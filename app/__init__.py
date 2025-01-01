from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config
import atexit

db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()

def cleanup_resources():
    """Cleanup function to be called on application exit"""
    try:
        db.session.remove()
    except Exception as e:
        print(f"Cleanup error: {e}")

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    # Register cleanup function
    atexit.register(cleanup_resources)

    # Register blueprints
    from app.routes import auth, dashboard, api
    app.register_blueprint(auth.bp)
    app.register_blueprint(dashboard.bp)
    app.register_blueprint(api.bp)

    # Initialize WebSocket after blueprints
    try:
        from app.websocket import init_websocket
        init_websocket(app)
    except Exception as e:
        print(f"WebSocket initialization failed: {e}")

    @app.route('/health')
    def health_check():
        return {'status': 'healthy'}, 200

    return app