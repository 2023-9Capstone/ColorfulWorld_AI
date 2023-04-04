from flask import request, jsonify
from flask_restx import Resource, Api, Namespace, fields

api = Namespace(
    name="post_img",
    description="이미지를 클라이언트에게 받아오기 위해 사용하는 API.",
)

#post 수신
@api.route('') 
class Recive(Resource):
    def post(self):
        image = request.files['image']
        idx =  request.form['index']
        #param = request.get_json()
        return {"filename" : image.filename, "index" : idx}