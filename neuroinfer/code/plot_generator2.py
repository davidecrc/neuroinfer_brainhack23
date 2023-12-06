from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def generate_plot(data):

    result_df, feature_names = ...
    cog_list, prior_list, x_target, y_target, z_target, radius, = parse_input_ars(data) #funzione(x,y,z,words,radius,probabiliteis)
    local_path, df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius = run_bayesian_analysis(cog_list, prior_list, x_target, y_target, z_target, radius, result_df, display_result=False)

    final_x, final_ticks, final_y = funzione(desired_output)
    # Perform your processing here...
    # For this example, let's create a simple plot
    plt.bar(words.split(','), [float(p) for p in probabilities.split(',')])
    plt.plot(final_x, final_y)

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Convert the image to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Send the base64 encoded image data as a response
    response = {
        'status': 'success',
        'message': 'Plot generated successfully',
        'image': img_base64
    }

    return response

def parse_input_ars(data): #funzione(x,y,z,words,radius,probabiliteis):

    # Extracting data from the form
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    words = data.get('words')
    radius = data.get('radius')
    probabilities = data.get('probabilities')
    
    if str(type(x)) != "float":
        try:
            x = float(x)
        except ValueError:
            print(f"Cannot convert {x} to a float")
            return None
    if str(type(y)) != "float":
        try:
            y = float(y)
        except ValueError:
            print(f"Cannot convert {y} to a float")
            return None
    if str(type(z)) != "float":
        try:
            z = float(z)
        except ValueError:
            print(f"Cannot convert {z} to a float")
            return None
    if str(type(radius)) != "float":
        try:
            radius = float(radius)
        except ValueError:
            print(f"Cannot convert {radius} to a float")
            return None
    if str(type(words)) != "list":
        try:
            words = list(words)
        except ValueError:
            print(f"Cannot convert {words} to a list")
            return None
    if str(type(probabilities)) != "list":
        try:
            probabilities = list(probabilities)
        except ValueError:
            print(f"Cannot convert {probabilities} to a list")
            return None
        if len(probabilities) == len(words):
            for i in probabilities:
                if i > 0 and i < 1:
                    pass
    return cog_list, prior_list, x_target, y_target, z_target, radius
if __name__ == '__main__':
    app.run(debug=True)
