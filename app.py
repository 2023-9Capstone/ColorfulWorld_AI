from flask import Flask, request, jsonify
from flask_restx import Api, Resource

app = Flask(__name__)
api = Api(app)

#get 송신
@api.route('/get/<string:name>') 
class Send(Resource):
    def get(self, name):
        return {"message" : "Welcome, %s!" % name}

#post 수신
@api.route('/post_image') 
class Recive(Resource):
    def post(self):
        param = request.get_json()
        return jsonify(param)

if __name__ == "__main__":
    app.run()