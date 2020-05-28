from flask import (Blueprint, render_template, make_response)

bp = Blueprint('deep_fake_model', __name__, url_prefix='/model')


@bp.route('/deep_fake_prediction', methods=['GET'])
def get_model_prediction():
    return render_template('homepage.html')


def load_model():
    print("loading model")
    
