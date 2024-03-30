from  flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


app = Flask(__name__)
# Load the pre-trained model
model = SimpleModel(input_size=784, output_size=10)  # Adjust input_size and output_size accordingly
model.load_state_dict(torch.load('model.path', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        # Save the image temporarily
        image_path = 'temp_image.png'
        image_file.save(image_path)
        # Preprocess the image
        input_tensor = preprocess_image(image_path)
        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        # Return the result as JSON
        result = {"predicted_class": predicted_class.item()}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
