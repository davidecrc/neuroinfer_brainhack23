from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import threading
import webbrowser
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from neuroinfer.code.plot_generator2 import generate_plot

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

PORT = 8000
SERVER_URL = f'http://localhost:{PORT}'
HOEMPAGE_URL = f'{SERVER_URL}/html/index.html'

@app.route('/', methods=['POST'])
def handle_post_request():
        response = generate_plot(request.json)
        return jsonify(response)

# Start the Flask app in a separate thread
def run_flask_app():
    app.run()


# Start the HTTP server in a separate thread
def run_http_server():
    handler = SimpleHTTPRequestHandler
    httpd = TCPServer(('', PORT), handler)
    httpd.serve_forever()


if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask_app)
    http_server_thread = threading.Thread(target=run_http_server)

    http_server_thread.start()
    # Open the HTML file in a web browser
    webbrowser.open_new(HOEMPAGE_URL)

    flask_thread.start()

    flask_thread.join()
    http_server_thread.join()
