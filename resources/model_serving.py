from flask import  request, jsonify, url_for
from flask_restx import Resource, Namespace
from data import colorize_image as CI
from PIL import Image
import numpy as np
from tempfile import NamedTemporaryFile
import cv2
from skimage import color
import os
from flask import current_app as app

api = Namespace(
    name="model_serving",
    description="모델을 가져오는 API.",
)

# Load the colorization model
colorModel = CI.ColorizeImageTorch(Xd=256)
colorModel.prep_net(None,'models/colorization_model.pth', False)


# Define a route for colorization
@api.route('', endpoint='uploaded_file')
class Colorize(Resource):
    def post(self):
        image = request.files['image']
        idx =  request.form['Intensity'] #인덱스 

        result = colorized(image)

        return jsonify({'url': result})
       
    
def colorized(img_file):

        #임시 파일 저장
        with NamedTemporaryFile(delete=False) as tmp:
            img_file.save(tmp.name)
            img = Image.open(tmp.name)
        
        # 이미지 처리 코드 작성
        file_path = tmp.name # 파일 경로 저장

        # Convert to grayscale
        colorModel.load_image(file_path)

        #img = Image.open(file)
        # 이미지를 numpy 배열로 변환
        img = img.convert('RGB')
        img = img.resize((256, 256)) #이미지 사이즈 조절(입력사이즈 256,256)
        img_array = np.array(img)
        # RGB 색상 공간을 Lab 색상 공간으로 변환
        img = color.rgb2lab(img_array)

        grid_size = 128
        h, w = img.shape[:2]
        cell_h, cell_w = int(h/grid_size), int(w/grid_size)
        cells = [img[r:r+cell_h, c:c+cell_w] for r in range(0, h, cell_h) for c in range(0, w, cell_w)]


        # 각 셀에서 가장 많은 색상으로 분류된 색상을 대표 색상으로 추출
        num_colors = 1  # 대표 색상 하나만 추출
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        rep_colors = []
        position = []
        for cell_idx, cell in enumerate(cells):
            pixel_colors = cell.reshape((-1, 3)).astype(np.float32)
            _, labels, centers = cv2.kmeans(pixel_colors, num_colors, None, criteria, 10, flags)
            
            rep_colors.append([centers[0][1],centers[0][2]])

            cell_row = cell_idx // (w // cell_w)  # 셀의 행 인덱스 계산
            cell_col = cell_idx % (w // cell_w)  # 셀의 열 인덱스 계산
            cell_position = (int(cell_row * cell_h ) , int(cell_col * cell_w)) # 셀의 위치 정보 저장
            position.append(cell_position)
        
        # initialize with no user inputs
        input_ab = np.zeros((2,256,256))
        mask = np.zeros((1,256,256))

        # Colorize image with hints
        for col, pos in zip(rep_colors, position):
            (input_ab,mask) = put_point(input_ab,mask,pos,1,col)

        # call forward
        img_out = colorModel.net_forward(input_ab,mask)

        # get mask, input image, and result in full resolution
        mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
        img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
        img_out_fullres = colorModel.get_img_fullres() # get image at full resolution
        

        # Return colorized image as response -> 이미지 파일 저장 및 이미지 url 제작
        result = Image.fromarray(img_out_fullres)
        img_path = 'image.png'
        filename = img_path
        result.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_url = url_for('static', filename='images/' + filename)

        os.remove(file_path) # 임시 파일 삭제
        return image_url

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
