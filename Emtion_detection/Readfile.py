import os
import cv2
import dlib
import numpy as np

# 根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Path
mypath = 'dataset/train'
savepath = 'FER_point_dataset/train'
allFileList = os.listdir(mypath)

# Dark img
dark_img = np.zeros((48,48), dtype=int)


for filename in os.listdir(r"./" + mypath):
    # 如果path = mypath+'/'+filename是資料夾(路徑)
    # 則進入資料夾
    if os.path.isdir(os.path.join(mypath, filename)):
        print("I'm a directory: " + filename)
        cv2.waitKey(1000)
        path = mypath + '/' + filename
        print('Path:', path)
        cv2.waitKey(1000)

        for file in os.listdir(r"./" + path):
            img = cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('img', img)
            # cv2.waitKey(200)

            # 找出特徵點位置
            d = dlib.rectangle(0, 0, 48, 48)
            shape = predictor(img, d)

            # 在空白圖片上 繪製68個特徵點
            for i in range(68):
                cv2.circle(dark_img,(shape.part(i).x,shape.part(i).y), 1,( 255, 255, 255), 1)
                cv2.circle(img,(shape.part(i).x,shape.part(i).y), 1,( 255, 255, 255), 1)
            
            # 儲存繪製圖片
            save = savepath + '/' + filename
            save = save + "/" + file
            cv2.imwrite(save, dark_img)
            # reset dark_img
            dark_img = np.zeros((48,48), dtype=int)

            # cv2.imshow('img', img)
            # cv2.waitKey(200)

            # print(img)              # For testing read file
            # print(np.shape(img))    # Check shape of img
    # 如果path是檔案
    elif os.path.isfile(mypath + filename):
        print(mypath)
        cv2.waitKey(1000)
        for file in os.listdir(r"./" + mypath):
            img = cv2.imread(mypath + "/" + file)
            cv2.imshow('img', img)
            cv2.waitKey(100)
    else:
        print('OH MY GOD !!')



array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,
    # directory_name is the folder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
        print(array_of_img)
