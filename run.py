from app import create_app
import socket
import os

def find_free_port(start_port=5000, max_attempts=100):
    """Find a free port starting from the given port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to create a socket with the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise OSError(f"No free ports found in range {start_port}-{start_port + max_attempts}")

app = create_app()

if __name__ == '__main__':
    try:
        # Find an available port
        port = find_free_port(start_port=5000)
        print(f"Starting server on port {port}")
        app.run(host='127.0.0.1', port=port, debug=True, use_reloader=True)
    except Exception as e:
        print(f"Failed to start server: {e}")
        # Try an alternative high-numbered port as a fallback
        try:
            fallback_port = 8080
            print(f"Attempting to start on fallback port {fallback_port}")
            app.run(host='127.0.0.1', port=fallback_port, debug=True, use_reloader=True)
        except Exception as e:
            print(f"Failed to start server on fallback port: {e}")
            raise