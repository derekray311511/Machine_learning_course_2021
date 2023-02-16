import cv2
import dlib
import time

# 載入分類器
face_cascade = cv2.CascadeClassifier('C:/Users/DDFish/anaconda3/envs/emotion02/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
# Timer
start_time = time.perf_counter()
# 讀取圖片
img = cv2.imread('face5.jpg')
img = cv2.resize(img, (640, 360))
# 轉成灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 偵測臉部
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.08,
    minNeighbors=5,
    minSize=(64, 64))
# 繪製人臉部份的方框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #給68特徵點辨識取得一個轉換顏色的frame
    landmarks_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #找出特徵點位置
    d = dlib.rectangle(x, y, x + w, y + h) 
    shape = predictor(landmarks_frame, d)

    #繪製68個特徵點
    for i in range(68):
        cv2.circle(img,(shape.part(i).x,shape.part(i).y), 1,( 0, 0, 255), 2)
        cv2.putText(img, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
#(0, 255, 0)欄位可以變更方框顏色(Blue,Green,Red)
# 顯示成果
cv2.namedWindow('img', cv2.WINDOW_NORMAL)  #正常視窗大小
cv2.imshow('img', img)                     #秀出圖片
cv2.imwrite("result2.jpg", img )           #保存圖片

# 計算執行時間(結束時間)
end_time = time.perf_counter()
process_time = end_time - start_time
print('process time:', round(process_time, 3))

cv2.waitKey(0)                             #等待按下任一按鍵
cv2.destroyAllWindows()                    #關閉視窗

