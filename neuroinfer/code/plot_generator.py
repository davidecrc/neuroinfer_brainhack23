from flask import jsonify
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def generate_plot(data):

    # Extracting data from the form
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    words = data.get('words')
    probabilities = data.get('probabilities')

    # Perform your processing here...
    # For this example, let's create a simple plot
    plt.bar(words.split(','), [float(p) for p in probabilities.split(',')])
    plt.show()

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

    return jsonify(response)
