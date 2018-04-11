# import modules
from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import codecs

# get path to saved model
sys.path.append(os.path.abspath("./model"))
from load import *

# initalize our flask app other variables
app = Flask(__name__)
global model, graph
model, graph = init()


# decoding an image from base64 into raw representation
def convertImageFromBase64(encImg):
    imgstr = re.search(b'base64,(.*)', encImg).group(1)
    with open('output.png', 'wb') as output:
        output.write(codecs.decode(imgstr, 'base64'))


# define route and functions
@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImageFromBase64(imgData)
    x = imread('output.png', 0)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    
    # perform the prediction
    with graph.as_default():    
        out = model.predict(x)
        response = np.array_str(np.argmax(out, axis=1))
        if response[1] == '0':
            res = "Not smiling"
        else:
            res = "Smiling"
        return res
