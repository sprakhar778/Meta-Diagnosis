from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from keras.models import load_model
from PIL import ImageOps, Image
import numpy as np
import secrets
import os

# Generate a random 24-byte (192-bit) secret key
random_secret_key = secrets.token_hex(24)

app = Flask(__name__)

app.secret_key = random_secret_key

def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image):- An image to be classified.
        model (tensorflow.keras.Model):- A trained machine learning model for image classification.
        class_names (list):- A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

@app.route('/classify', methods=['POST'])
def classify_image():
    # load classifier
    model = load_model('./model/pneumonia_classifier.h5')

    # load class names
    with open('./model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    # get image file
    file = request.files['file']

    # read image
    image = Image.open(file).convert('RGB')

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # store classification result in session
    session['class_name'] = class_name
    session['conf_score'] = int(conf_score * 1000) / 10
    # redirect to pneumonia page
    return redirect(url_for('pneumonia'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')


@app.route('/pneumonia')
def pneumonia():
    class_name = session.pop('class_name', 'Unknown')
    conf_score = session.pop('conf_score', 'Unknown')
    return render_template('pneumonia.html', class_name=class_name, conf_score=conf_score)

if __name__ == '__main__':
    app.run(debug=True)
