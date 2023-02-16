
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential      # 啟動NN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D          # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D    # 池化
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten, Dropout        
from tensorflow.keras.layers import Dense           # Fully-Connected Networks
from tensorflow.keras.layers import RNN, LSTM, SimpleRNN
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import datetime
import os
import glob

import dlib
import cv2
from keras_video import VideoFrameGenerator


# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict',help='train/predict')
ap.add_argument('--weights', default='run/train-face/RNN53epochs/weights.53-0.79.h5', help='weights.h5 path')
ap.add_argument('--source', default='smellgood.mp4',help='source')
ap.add_argument('--epochs', type=int, default=30)
opt = ap.parse_args()


# some global params
SIZE = (48, 48)
CHANNELS = 1
NBFRAME = 6
BS = 8
EPOCHS = opt.epochs

mode = opt.mode
weights_dir = opt.weights
source = opt.source
if mode == 'train':
    print("========================================")
    print("Mode =", mode)
    print("Weights =", weights_dir)
    print("Epochs =", EPOCHS)
    print("========================================")

    # Build and compile model
    cnn = load_model('run/train-face/89epochs_VGG/model_keras.h5')
    cnn.load_weights('run/train-face/89epochs_VGG/best_weights.h5')
    # cnn = load_model('run/train-face/89epochs_VGG/model_keras.h5')
    # cnn.load_weights('run/train-face/89epochs_VGG/best_weights.h5')
    cnn = Model(inputs=cnn.input, outputs=cnn.layers[-2].output)

    model = Sequential()

    model.add(TimeDistributed(cnn, input_shape=(NBFRAME, 48, 48, 1)))
    model.add(SimpleRNN(units=35, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))

    for layer in model.layers[:-2]:
        print('Trainable layer:', layer.trainable)
        layer.trainable = False
        print('Trainable layer:', layer.trainable)

    # Compile the model
    # optimizers = Adam(learning_rate=0.001)
    optimizers = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizers, 
                metrics='accuracy')
    model.summary()

    # Save model structure picture
    plot_model(model, show_shapes=True, to_file='run/train-face/epochs/model.png')



    # use sub directories names as classes
    classes = [i.split(os.path.sep)[1] for i in glob.glob('Video(face)/*')]
    classes.sort()
    print('Class names:', classes)

    # pattern to get videos and classes
    glob_pattern='Video(face)/{classname}/*.avi'

    # for data augmentation
    data_aug = ImageDataGenerator(
                        rescale=1./255,
                        zoom_range=.1,
                        horizontal_flip=True,
                        rotation_range=10,
                        width_shift_range=.2,
                        height_shift_range=.2)

    # Create video frame generator
    train = VideoFrameGenerator(
                        classes=classes, 
                        glob_pattern=glob_pattern,
                        nb_frames=NBFRAME,
                        split=.2, 
                        shuffle=True,
                        batch_size=BS,
                        target_shape=SIZE,
                        nb_channel=CHANNELS,
                        transformation=data_aug,
                        use_frame_cache=True)

    valid = train.get_validation_generator()

    import keras_video.utils
    # keras_video.utils.show_sample(train)


    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [ReduceLROnPlateau(factor=0.5, verbose=1, min_lr=0.00001),
                 ModelCheckpoint(
                     'run/train-face/epochs/chkp/weights.{epoch:02d}-{val_accuracy:.2f}.h5',
                     save_best_only=False,
                     save_weights_only=True,
                     verbose=1),
                 TensorBoard(log_dir=log_dir, histogram_freq=1)]

    model_info = model.fit(
                    train,
                    validation_data=valid,
                    verbose=1,
                    epochs=EPOCHS,
                    callbacks=callbacks
    )
    model.save('run/train-face/epochs/model_keras.h5')
    model.save_weights('run/train-face/epochs/last_weights.h5')

    # 繪製訓練 & 驗證的準確率值
    plt.figure()
    plt.grid()
    plt.plot(model_info.history['accuracy'])
    plt.plot(model_info.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('run/train-face/epochs/History_acc.png')

    # 繪製訓練 & 驗證的損失值
    plt.figure()
    plt.grid()
    plt.plot(model_info.history['loss'])
    plt.plot(model_info.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('run/train-face/epochs/History_loss.png')
    plt.show()


elif mode == 'predict':
    # 載入模型
    model = load_model('run/train-face/RNN38epochs_VGG/model_keras.h5')
    model.load_weights(weights_dir)
    # 選擇輸入 0: 第一隻攝影機 mp4: Video
    cap = cv2.VideoCapture(source)
    # 儲存影片結果
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('run/results/output_face.mp4',fourcc, 24.0, (640, 360))
    # out = cv2.VideoWriter('run/results/output_face.mp4',fourcc, 4.0, (256, 256))
    # 編號對應情緒
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    # 取得預設的臉部偵測器
    detector = dlib.get_frontal_face_detector()
    # 根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 當攝影機打開時，對每個frame進行偵測
    skip = 4 # 決定偵測到幾次臉辨識一次情緒
    while(cap.isOpened()):
        # 計算執行時間(起始時間)
        start_time = time.perf_counter()
        # 讀出frame資訊
        ret, frame = cap.read()
        # 改變輸入影像大小
        frame = cv2.resize(frame, (640, 360))
        # 偵測人臉
        face_rects, scores, idx = detector.run(frame, 0)
        # 取出偵測的結果
        for i, d in enumerate(face_rects):
            x1 = d.left()
            y1 = d.top()
            x2 = d.right()
            y2 = d.bottom()
            text = "%2.2f(%d)" % (scores[i], idx[i])
            # 繪製出偵測人臉的矩形範圍
            cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 1, cv2. LINE_AA)
            # 彩色BGR to 灰階
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 裁切圖片(Crop)
            cropped_frame = gray_frame[y1:y2, x1:x2]
            # 升維
            try:
                cropped_frame = np.expand_dims(np.expand_dims(np.expand_dims(cv2.resize(cropped_frame, (48, 48), cv2.INTER_NEAREST), -1), 0), 0)
            except:
                break
            if skip >= 4 or i > 0:   # 決定偵測到幾次臉辨識一次情緒
                # 情緒識別
                prediction = model.predict(cropped_frame)
                skip = 0
            maxindex = int(np.argmax(prediction))
            emotion_prob = round(prediction[0][maxindex], 2)
            emotion_text = emotion_dict[maxindex] + " " + str(emotion_prob)
            cv2.putText(frame, emotion_text, (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            # 如果按下ESC键，就退出
            if cv2.waitKey(10) == 27:
                break
            

        # 輸出到畫面
        cv2.imshow("Emotion Detection", frame)
        # 儲存結果
        out.write(frame)
        # 計算執行時間(結束時間)
        end_time = time.perf_counter()
        process_time = end_time - start_time
        print('process time:', round(process_time, 3))
        skip += 1
        # 如果按下ESC键，就退出
        if cv2.waitKey(10) == 27:
            break

    # 釋放記憶體
    cap.release()
    # 關閉所有視窗
    cv2.destroyAllWindows()