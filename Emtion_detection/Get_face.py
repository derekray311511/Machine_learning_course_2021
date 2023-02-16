import os
import cv2
import dlib
import numpy as np

# 臉部識別
detector = dlib.get_frontal_face_detector()

# Path
mypath = 'Video_data'
savepath = 'Video_data(face)/FACE'
allFileList = os.listdir(mypath)


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
            # cv2.imshow('img', img)
            # cv2.waitKey(200)

            # 找出特徵點位置
            face_rects, scores, idx = detector.run(img, 0)

            # 取出偵測的結果
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                # 擷取臉部 
                cropped_img = img[y1:y2, x1:x2]
                try:
                    cropped_img = cv2.resize(cropped_img, (48, 48), cv2.INTER_NEAREST)
                except:
                    break
                cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 255, 0), 1, cv2. LINE_AA)

                # 儲存繪製圖片
                save = savepath + '/' + filename
                save = save + "/" + file
                cv2.imwrite(save, cropped_img)
            
            cv2.imshow('img', img)
            cv2.waitKey(50)

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
