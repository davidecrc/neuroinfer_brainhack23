from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import threading
import webbrowser
from flask import Flask, request, jsonify
from flask_cors import CORS
from neuroinfer.code.rendering_manager import main_analyse_and_render, create_mask_region, update_overlay

# CORS (Cross-Origin Resource Sharing):
# - CORS is a browser security feature controlling access to resources from different domains.
# - It prevents potentially unsafe cross-origin requests while allowing legitimate interactions.
# - Correct CORS headers must be set on servers to determine whether requests from specific origins are allowed.

# Flask:
# - Flask is a lightweight and flexible Python web framework.
# - It enables rapid development of web applications with minimal boilerplate code.
# - Following the WSGI standard, Flask is compatible with various web servers.
# - It provides features for routing, handling HTTP requests and responses, and easy integration with other libraries.
# - Known for simplicity, flexibility, and extensibility, Flask is popular for building web applications and APIs in Python.

# Create a Flask app instance and enable Cross-Origin Resource Sharing (CORS)
app = Flask(__name__)
CORS(app)

# Define constants for the server configuration
PORT = 8031
SERVER_URL = f'http://localhost:{PORT}'
HOMEPAGE_URL = f'{SERVER_URL}/html/index.html'


# Define a route for handling POST requests at the root path
@app.route('/', methods=['POST'])
def handle_post_request():
    # Extract the JSON data from the POST request
    dict_request = request.json

    # Get the keys from the request's JSON data
    form_keys = dict_request.keys()
    print(form_keys)
    print(dict_request)

    # Check if only one key is present in the request
    if 'func' in form_keys and dict_request['func'] == "update_mask":
        # If the only key is 'brainRegion', attempt to create a mask for the specified brain region
        response = create_mask_region(dict_request['atlas'], dict_request['brainRegion'], dict_request['smooth'])
    # If 'combination_bool' and 'file_list' keys are present, send them to update_overlay
    elif 'func' in form_keys and dict_request['func'] == "update_overlays":
        print(dict_request['combination_bool'])
        print(dict_request['file_list'])
        response = update_overlay(dict_request['combination_bool'], dict_request['file_list'])
    elif 'func' in form_keys and dict_request['func'] == "do_analysis":
        # If multiple keys are present, perform the main analysis and rendering
        response = main_analyse_and_render(dict_request)
    elif 'func' in form_keys and dict_request['func'] == "load_results":
        response = load_results(loadfile)

    # Return the response as JSON
    return jsonify(response)



# Function to start the Flask app in a separate thread
def run_flask_app():
    # Start the Flask app on the specified port
    app.run()

# Function to start the HTTP server in a separate thread
def run_http_server():
    # Use SimpleHTTPRequestHandler to handle HTTP requests
    handler = SimpleHTTPRequestHandler

    # Create a TCP server bound to the specified port
    httpd = TCPServer(('', PORT), handler)

    # Start serving HTTP requests indefinitely
    httpd.serve_forever()


# Entry point of the script
if __name__ == '__main__':
    # Create threads for running the Flask app and HTTP server
    flask_thread = threading.Thread(target=run_flask_app)
    http_server_thread = threading.Thread(target=run_http_server)

    # Start the HTTP server and open the homepage in a web browser
    http_server_thread.start()
    webbrowser.open_new(HOMEPAGE_URL)

    # Start the Flask app
    flask_thread.start()

    # Wait for both threads to complete before exiting the script
    flask_thread.join()
    http_server_thread.join()
