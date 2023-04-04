from flask import request, jsonify
from flask_restx import Resource, Api, Namespace, fields

api = Namespace(
    name="get_img",
    description="클라이언트가 변경된 이미지를 서버에서 받아오기 위해 사용하는 API.",
)


#get 송신
@api.route('/<string:name>') 
class Send(Resource):
    def get(self, name):
        return {"message" : "Welcome, %s!" % name}