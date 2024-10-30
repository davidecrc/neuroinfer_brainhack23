from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import threading
import webbrowser
from flask import Flask, request, jsonify
from flask_cors import CORS
from neuroinfer.code.python.rendering_manager import (
    main_analyse_and_render,
    create_mask_region,
    update_overlay,
    load_results,
)

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
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Define constants for the server configuration
PORT = 8000
MAX_TRIES = 10  # maximum number of ports to try


# Define a route for handling POST requests at the root path
@app.route("/", methods=["POST"])
def handle_post_request():
    # Extract the JSON data from the POST request
    dict_request = request.json

    # Get the keys from the request's JSON data
    form_keys = dict_request.keys()
    print(form_keys)
    print(dict_request)

    # Check if only one key is present in the request
    if "func" in form_keys and dict_request["func"] == "update_mask":
        # If the only key is 'brainRegion', attempt to create a mask for the specified brain region
        response = create_mask_region(
            dict_request["atlas"], dict_request["brainRegion"], dict_request["smooth"]
        )
    # If 'combination_bool' and 'file_list' keys are present, send them to update_overlay
    elif "func" in form_keys and dict_request["func"] == "update_overlays":
        print(dict_request["combination_bool"])
        print(dict_request["file_list"])
        response = update_overlay(
            dict_request["combination_bool"], dict_request["file_list"]
        )
    elif "func" in form_keys and dict_request["func"] == "do_analysis":
        # If multiple keys are present, perform the main analysis and rendering
        response = main_analyse_and_render(dict_request)
    elif "func" in form_keys and dict_request["func"] == "load_results":
        response = load_results(dict_request["loadfile"])

    # Return the response as JSON
    return jsonify(response)


# Function to start the Flask app in a separate thread
def run_flask_app():
    # Start the Flask app on the specified port
    app.run()


# Function to start the HTTP server in a separate thread
def run_http_server():
    handler = SimpleHTTPRequestHandler
    global PORT  # Make PORT a global variable so it can be updated

    for i in range(MAX_TRIES):
        try:
            httpd = TCPServer(("", PORT + i), handler)
            # Update the port number globally
            global actual_port
            actual_port = PORT + i
            print(f"Serving on port {actual_port}")
            # Start serving HTTP requests indefinitely
            httpd.serve_forever()
            break
        except OSError as e:
            if e.errno == 98:
                print(f"Port {PORT + i} already in use. Trying next port...")
            else:
                raise


# Entry point of the script
if __name__ == "__main__":
    # Create threads for running the Flask app and HTTP server
    flask_thread = threading.Thread(target=run_flask_app)
    http_server_thread = threading.Thread(target=run_http_server)

    # Start the HTTP server thread
    http_server_thread.start()

    # Wait a bit to ensure the server is up and running
    http_server_thread.join(1)  # Adjust the wait time as needed

    # Update HOMEPAGE_URL with the actual port number
    homepage_url = f"http://localhost:{actual_port}/neuroinfer"
    print(f"Open the following URL in your browser: {homepage_url}")

    # Open the updated homepage URL in a web browser
    webbrowser.open_new(homepage_url)

    # Start the Flask app
    flask_thread.start()

    # Wait for both threads to complete before exiting the script
    flask_thread.join()
    http_server_thread.join()
