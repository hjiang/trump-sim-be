# coding: utf-8
import sys

import leancloud
from flask import Flask, jsonify, request
from flask import render_template
from flask_sockets import Sockets
from flask_cors import CORS

from fastai2.learner import load_learner

app = Flask(__name__)
CORS(app)

sockets = Sockets(app)

trump = load_learner('model.pkl')


@app.route('/')
def index():
    response = jsonify({'status': 'ok'})
    response.status_code = 200
    return response

@app.route('/api/1.0/classify-image', methods=['POST'])
def classify():
    image = request.files['image']
    res = trump.predict(image.read())
    response = jsonify({'result': res[0]})
    response.status_code = 200
    return response
