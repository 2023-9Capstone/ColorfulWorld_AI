from flask import  request, jsonify, url_for, send_file
from flask_restx import Resource, Namespace
from data import colorize_image as CI
from PIL import Image
import numpy as np
from tempfile import NamedTemporaryFile
import cv2
from skimage import color
import os
from flask import current_app as app
import pymysql
import logging
import shutil
import pickle

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
 
        result, code = colorized(image,idx)
        return jsonify({'url': result}) #
        #return jsonify({'url': result, 'error' : code}) #code 0은 혼동선 없음, 1은 있음

    
def colorized(img_file, intensity):
        tens = [13,26,39,13,26,39,13,26,39] #0,1,2,3
        
        #임시 파일 저장
        with NamedTemporaryFile(delete=False) as tmp:
            img_file.save(tmp.name)
            img = Image.open(tmp.name)
        
        # 이미지 처리 코드 작성
        file_path = tmp.name # 파일 경로 저장

        # Convert to grayscale
        colorModel.load_image(file_path)

        # 이미지를 numpy 배열로 변환
        img = img.convert('RGB')
        img = img.resize((256, 256)) #이미지 사이즈 조절(입력사이즈 256,256)
        img_array = np.array(img)
        # RGB 색상 공간을 Lab 색상 공간으로 변환
        img = color.rgb2lab(img_array)

        #grid_size = 128
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
        compare_colors = []
        for cell_idx, cell in enumerate(cells):
            pixel_colors = cell.reshape((-1, 3)).astype(np.float32)
            _, labels, centers = cv2.kmeans(pixel_colors, num_colors, None, criteria, 10, flags)

            # if centers[0][1]<0 and centers[0][2]<0 :
            #     compare_colors.append([int(centers[0][0]/5),int(centers[0][1]/13-1),int(centers[0][2]/13-1)])
            # elif centers[0][1]>=0 and centers[0][2]<0 :
            #     compare_colors.append([int(centers[0][0]/5),int(centers[0][1]/13),int(centers[0][2]/13-1)])
            # elif centers[0][1]<0 and centers[0][2]>=0:
            #     compare_colors.append([int(centers[0][0]/5),int(centers[0][1]/13-1),int(centers[0][2]/13)])
            # else :
            compare_colors.append([int(centers[0][0]/5),int(centers[0][1]/13),int(centers[0][2]/13)])
            rep_colors.append([centers[0][0],centers[0][1],centers[0][2]])


            cluster_idx = np.argmin(np.linalg.norm(centers - rep_colors[-1], axis=1))
            cluster_pixels = pixel_colors[labels.flatten() == cluster_idx]
            #import pdb;pdb.set_trace();

            closest_pixel = cluster_pixels[np.argmin(np.linalg.norm(cluster_pixels - rep_colors[-1], axis=1))]
            closest_pixel_pos = np.where((pixel_colors == closest_pixel).all(axis=1))
            x, y = (closest_pixel_pos[0][0]+1)//4,closest_pixel_pos[0][0]%4

            cell_idx_x = cell_idx % grid_size
            cell_idx_y = cell_idx // grid_size

            pos_x = cell_idx_x * cell_w + y
            pos_y = cell_idx_y * cell_h + x

            position.append([pos_y,pos_x])
        
        compare = list(set(map(tuple, compare_colors)))   

        import pickle
        import pprint

        if int(intensity)<=2 :
            with open('data/CVD_P_change_13_0609.pkl', 'rb') as file:
                loaded_list = pickle.load(file)
        elif int(intensity)<=5 :
            with open('data/CVD_D_change_13.pkl', 'rb') as file:
                loaded_list = pickle.load(file)
        elif int(intensity)<=8 :
            with open('data/CVD_T_change_13.pkl', 'rb') as file:
                loaded_list = pickle.load(file)


        cnt = 0
        line = []
        color_line=[]
        for idx_i, cvd in enumerate(loaded_list):
            for com in compare:
                color_line=[]
                if cvd[0][0]==com[0] and  cvd[0][1]==com[1] and cvd[0][2]==com[2]:
                    for cvv in cvd:
                        for cm in compare:
                            if cvv[0]==cm[0] and  cvv[1]==cm[1] and cvv[2]==cm[2]:
                                color_line.append([cm[0],cm[1],cm[2]])
                                compare.remove(cm)
        
            if len(color_line)>=2 :
                line.append(color_line)
                cnt += len(color_line)
                color_line = []
       

        for line_confuse in line :
            confuse = []
            ccc = []
            print(line_confuse)
            for col in line_confuse:
                count = 0
                for com, rep in zip(compare_colors, rep_colors):
                    if col[0] == com[0] and col[1]==com[1] and col[2]==com[2]:
                        count += 1
                if count > 0 :
                    ccc.append(count)
                    confuse.append([col[0], col[1], col[2]])
            if len(ccc)>=2:
                print(confuse)
                print(ccc)
                idx_max = max(ccc)
                idx_m = ccc.index(idx_max)
                print(idx_m)
                for col_li, c in zip(confuse, ccc) :
                    if c !=idx_max:
                        for com, rep in zip(compare_colors, rep_colors):
                            if com[1] == 0 and com[2] == 0 :
                                a = 0
                            elif col_li[0] == com[0] and col_li[1]==com[1] and col_li[2]==com[2] and (confuse[idx_m][1]*com[1]<=0 or confuse[idx_m][2]*com[2]<=0 )  :
                                #rep[0] -= 10
                                if confuse[idx_m][1]>0:
                                    rep[1] -= tens[int(intensity)]
                                else :
                                    rep[1] += tens[int(intensity)]
                                if confuse[idx_m][2]>0:
                                    rep[2] -= tens[int(intensity)]
                                else :
                                    rep[2] += tens[int(intensity)]
        have_line = 0
        if have_line >=0 :
            # initialize with no user inputs
            input_ab = np.zeros((2,256,256))
            mask = np.zeros((1,256,256))

            # Colorize image with hints
            for col, pos in zip(rep_colors, position):
                (input_ab,mask) = put_point(input_ab,mask,pos,1,col[1:])

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
            result_url = url_for('static', filename='images/' + filename)

            
            return result_url, 1
        
        else :    
            return 'nothing' , 0

        

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

