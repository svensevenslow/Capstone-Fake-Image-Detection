import os

from flask import (Flask, render_template,g)
from flaskr import meta
from flaskr import model, deep_fake_model

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    app.register_blueprint(meta.bp)
    app.register_blueprint(model.bp)
    app.register_blueprint(deep_fake_model.bp)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def home():
        return render_template('homepage.html')

    return app

