from flask import (Blueprint, request, make_response)
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import RMSprop
import numpy as np
from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance
bp = Blueprint('model', __name__, url_prefix='/model')


@bp.route('/prediction', methods=['GET'])
def get_model_prediction():
    K.clear_session()
    loaded_model = load_model()

    path = "uploads/tmp.jpg"
    quality = 90

    image_to_predict = np.array(array(convert_to_ela_image(path, quality).resize((128, 128))).flatten() / 255)
    image_to_predict = image_to_predict.reshape(-1, 128, 128, 3)
    result = loaded_model.predict(image_to_predict)
    #print(result.astype(np.int32))
    fake = result[0][1]*100
    print(fake)
    return fake
    #return make_response({"fake":result[0][1], "real":result[0][0]},200)


def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = 'uploads/' + filename.split('/')[-1].split('.')[0] + '.resaved.jpg'

    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return ela_im


def load_model():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")
    optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    loaded_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return loaded_model
