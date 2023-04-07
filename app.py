from flask import Flask, request, jsonify
from flask_restx import Api, Resource #flask restful
from flask_cors import CORS
from resources.post_image import api as post_namespace
from resources.get_image import api as get_namespace
from resources.model_serving import api as model_namespace

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "http://172.16.65.77"}})

api = Api(
    app,
    version = '0.1',
    title = "Colorful World deep learing Server",
    terms_url = "/"
    )

@app.route('/')
def index():
    app.logger.info('Web connection established.')
    return 'Hello, World!'

api.add_namespace(post_namespace, path="/post")
api.add_namespace(get_namespace, path="/get")
api.add_namespace(model_namespace, path="/colorization")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)