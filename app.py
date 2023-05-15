from flask import Flask, request, jsonify
from flask_restx import Api, Resource #flask restful
from flask_cors import CORS
from resources.post_image import api as post_namespace
from resources.get_image import api as get_namespace
from resources.model_serving import api as model_namespace
import os
import pymysql

app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version = '0.1',
    title = "Colorful World deep learing Server",
    terms_url = "/"
    )

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'images')

app.config['MYSQL_HOST'] = 'localhost'  # MySQL 호스트
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'  # MySQL 사용자 이름
app.config['MYSQL_PASSWORD'] = 'rbtns0710'  # MySQL 비밀번호
app.config['MYSQL_DB'] = 'COLORBLIND'  # MySQL 데이터베이스 이름

#mysql = MySQL(app)

@app.route('/')
def index():
    connection = pymysql.connect(
        host=app.config['MYSQL_HOST'],
        port=app.config['MYSQL_PORT'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        db=app.config['MYSQL_DB']
    )

    # 쿼리 실행 예시
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM COLORBLIND.protanopin')
    result = cursor.fetchall() 
    #print(result)
    # 결과 처리 예시

    output = ''
    for row in result:
        output += f'{row[0]}\n'
    # 연결 종료
    connection.close()

    return output

api.add_namespace(post_namespace, path="/post")
api.add_namespace(get_namespace, path="/get")
api.add_namespace(model_namespace, path="/colorization")

                        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5111, debug=True)
    index()