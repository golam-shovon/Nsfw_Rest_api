import urllib
import pandas as pd
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import keras
import json
import tensorflow as tf
from flask import request as fr
import request
import operator
from flask import redirect
import logging
import os
app = flask.Flask(__name__)
nsfw_model = None

graph=None
handler = logging.FileHandler('app.log')  # errors logged to this file
handler.setLevel(logging.ERROR)  # only log errors and above
app.logger.addHandler(handler)

@app.before_first_request
def load_model():
    global nsfw_model,graph
    nsfw_model = None
    nsfw_model = keras.models.load_model('nsfw.299x299.h5')
    graph = tf.get_default_graph()


def prepare_image(image, image_size):
    loaded_images =[]
    image = keras.preprocessing.image.load_img(image, target_size=image_size)
    image = keras.preprocessing.image.img_to_array(image)
    image /= 255
    loaded_images.append(image)
    return np.asarray(loaded_images)

@app.route("/predict", methods=["GET"])
def predict():
    domain = fr.args.get('domain_url')
    id = fr.args.get('id')
    url = fr.args.get('image_url')
    urllib.request.urlretrieve(url, 'pic.jpg')
    image = 'pic.jpg'

    image_size = (299, 299)
    image = prepare_image(image, image_size)
    categories = ["drawings", "neutral", "sexy", "hentai", "porn"]
    with graph.as_default():
        model_preds = nsfw_model.predict(image)
        preds = np.argsort(model_preds, axis=1).tolist()

    predjson=json.dumps(preds)
    check = ['hentai', 'porn']
    probs = []
    for i, single_preds in enumerate(preds):
        single_probs = []
        for j, pred in enumerate(single_preds):
            single_probs.append(model_preds[i][pred])
            preds[i][j] = categories[pred]

    probs.append(single_probs)

    result = dict(zip(categories, single_probs))
    result_check = pd.Series(result)
    result['imageid'] = id
    resultjson=pd.Series(result).to_json(orient='index')
    cause = max(result_check.items(), key=operator.itemgetter(1))
    if 'hentai' or 'porn' in str(cause):
        is_safe = False
    else:
        is_safe = True
    os.remove('pic.jpg')
    #With open('requestlog.txt', 'a') as f:
       ##f.write('\n')
        ##f.write('\n')''
    redirect_url = domain + '/contentdetector/response?' + 'id=' + id + '&is_safe=' + str(is_safe) + '&reason=' + str(cause)
    #return json.dumps(cause)
    return redirect(redirect_url)
 

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
