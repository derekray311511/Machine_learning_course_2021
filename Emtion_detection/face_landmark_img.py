# -*- coding: utf-8 -*-
import dlib
import cv2
import imutils
import time

# Timer
start_time = time.perf_counter()
#讀取圖片
img = cv2.imread('human.jpg')
# img = cv2.resize(img, (640, 360))
#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

start = time.time()
#偵測人臉
face_rects, scores, idx = detector.run(img, 0)
end = time.time()

#取出偵測的結果
for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    text = " %2.2f ( %d )" % (scores[i], idx[i])

    #繪製出偵測人臉的矩形範圍
    cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 255, 0), 2, cv2. LINE_AA)

    #標上人臉偵測分數與人臉方向子偵測器編號
    cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX,
    0.7, ( 255, 255, 255), 1, cv2. LINE_AA)

    #給68特徵點辨識取得一個轉換顏色的frame
    landmarks_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #找出特徵點位置
    shape = predictor(landmarks_frame, d)

    #繪製68個特徵點
    # for i in range(68):
    #     cv2.circle(img,(shape.part(i).x,shape.part(i).y), 1,( 0, 0, 255), 2)
    #     cv2.putText(img, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)  #正常視窗大小
cv2.imshow('img', img)                     #秀出圖片
cv2.imwrite("result2.jpg", img )           #保存圖片

# 計算執行時間(結束時間)
end_time = time.perf_counter()
process_time = end_time - start_time
print('process time:', round(process_time, 3))

part_process_time = end - start
print('part process time:', round(part_process_time, 3))

cv2.waitKey(0)                             #等待按下任一按鍵
cv2.destroyAllWindows()                    #關閉視窗

