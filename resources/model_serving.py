from flask import Flask, request, jsonify
from flask_restx import Resource, Api, Namespace, fields
from data import colorize_image as CI
from PIL import Image
import numpy as np
from io import BytesIO
from tempfile import NamedTemporaryFile

api = Namespace(
    name="model serving",
    description="모델을 가져오는 API.",
)

# Load the colorization model
colorModel = CI.ColorizeImageTorch(Xd=256)
colorModel.prep_net(None,'models/colorization_model.pth', False)


# Define a route for colorization
@api.route('')
class Colorize(Resource):
    
    def post(self):
        image = request.files['image']
        idx =  request.form['index']
        return  colorized()

#Adding user inputs 
# lab 컬러러 중중 ab 사용
def put_point(input_ab,mask,loc,p,val):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       half-patch size
    # val         2 tuple      (a,b) value of user input (color)
    input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
    mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
    return (input_ab,mask)
    
def colorized():
        # Load image from the request
        img_file = request.files['image']

        with NamedTemporaryFile(delete=False) as tmp:
            img_file.save(tmp.name)
            img = Image.open(tmp.name)
        # 이미지 처리 코드 작성
        file_path = tmp.name # 파일 경로 저장

        #img_file = img_file.read()
        #img = Image.open(BytesIO(img_file))
        #img = np.array(Image.open(BytesIO(img_file)))
        #img = img_file.convert('RGBA')

        # Convert to grayscale
        colorModel.load_image(file_path)

        # initialize with no user inputs
        input_ab = np.zeros((2,256,256))
        mask = np.zeros((1,256,256))

        # Colorize image with hints
        # add a blue point in the middle of the imag
        (input_ab,mask) = put_point(input_ab,mask,[150,160],3,[60,61])

        # call forward
        img_out = colorModel.net_forward(input_ab,mask)

        # get mask, input image, and result in full resolution
        mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
        img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
        img_out_fullres = colorModel.get_img_fullres() # get image at full resolution

        # Return colorized image as response -> 수정에 따라 사용할 예정
        response = jsonify({'result': img_out_fullres})
        return response
    