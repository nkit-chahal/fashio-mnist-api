
import os
import requests
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

print(tf.__version__)

#Load the pretrained model
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("fashion_model_flask.h5")



app = Flask(__name__)

#classify_image function
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    upload_dir = "uploads/"
    image = plt.imread(upload_dir + img_name)
    
    #list of class names 
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    prediction = model.predict([image.reshape(1, 28*28)])

    
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})


app.run(port=120, debug=False)