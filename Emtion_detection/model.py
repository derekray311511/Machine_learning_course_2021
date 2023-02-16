# Import Dependencies
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential      # 啟動NN
from tensorflow.keras.layers import Conv2D          # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D    # 池化
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten, Dropout        
from tensorflow.keras.layers import Dense           # Fully-Connected Networks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau

import h5py
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import datetime

import dlib
import cv2
import imutils
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict',help='train/predict')
ap.add_argument('--weights', default='weights/model.h5', help='model.h5 path')
ap.add_argument('--source', default='Trump.mp4',help='source')
ap.add_argument('--epochs', type=int, default=30)
opt = ap.parse_args()

# Parameters
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = opt.epochs

mode = opt.mode
weights_dir = opt.weights
source = opt.source
if mode == 'train':
    print("========================================")
    print("Mode =", mode)
    print("Weights =", weights_dir)
    print("Epochs =", num_epoch)
    print("========================================")


# Build and compile model
# initializing CNN

# Small Structure epoch186

# model = Sequential()

# model.add(Conv2D(7, kernel_size=(3, 3), padding='same', input_shape = (48, 48, 1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(14, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(28, kernel_size=(5, 5), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(112, kernel_size=(5, 5), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(units = 84, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units = 7, activation='softmax'))

# optimizers = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')


# Small Structure epoch111

# model = Sequential()

# model.add(Conv2D(8, kernel_size=(5,5), padding='same', input_shape = (48, 48, 1), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))

# model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, kernel_size=(5, 5), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(units = 84, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units = 7, activation='softmax'))

# optimizers = Adam(learning_rate=0.001)
# model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')


# model = Sequential()

# model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape = (48, 48, 1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(AveragePooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(units = 4096))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(units = 4096))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(units = 7, activation='softmax'))

# optimizers = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')

model = Sequential()

model.add(Conv2D(8, kernel_size=(5,5), strides = 1, padding='same', input_shape = (48, 48, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3), padding='same'))
model.add(Dropout(0.25))
# Second convolutional layer
model.add(Conv2D(16, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3), padding='same'))
model.add(Dropout(0.25))
# Third convolutional layer
model.add(Conv2D(32, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3), padding='same'))
model.add(Dropout(0.25))
# Forth convolutional layer
model.add(Conv2D(64, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3), padding='same'))
model.add(Dropout(0.25))
# fifth convolutional layer
model.add(Conv2D(128, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3), padding='same'))
model.add(Dropout(0.25))
# sixth convolutional layer
model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', activation = 'relu'))
model.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
model.add(Dropout(0.25))
# 將 feature maps 攤平放入一個向量中
model.add(Flatten())
# Fully-Connected Networks
model.add(Dense(units = 200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 7, activation = 'softmax'))
# # Compiling the CNN 
optimizers = SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics='accuracy')


# Training mode

if mode == 'train':
    # Show model info
    model.summary()
    # Save model structure picture
    keras.utils.plot_model(model, show_shapes=True, to_file='run/train-face/epochs/model.png')
    # initial weights
    # model.load_weights("run/train-face/epochs/best_weights_40.h5")  
    checkpoint = ModelCheckpoint("run/train-face/epochs/best_weights.h5", 
        monitor='val_accuracy', verbose=1, save_best_only=True, 
        mode='auto', save_freq='epoch', save_weights_only=True)
    EarlyStop = EarlyStopping(monitor='loss', patience=10)
    adapt_lr = ReduceLROnPlateau(factor=0.1, patience=5, verbose=1, min_lr=0.000001)
    # TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    print('\ntensorboard --logdir logs/fit\n')
    
    # Data input and data augmentation
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2,
                                       zoom_range = 0.2, horizontal_flip = True,
                                       rotation_range=10, brightness_range=(0.2, 0.8),
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       zca_whitening=False, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory(
        'dataset/train', 
        target_size = (48, 48), 
        batch_size = batch_size, 
        color_mode="grayscale", 
        class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory(
        'dataset/test', 
        target_size = (48, 48), 
        batch_size = batch_size, 
        color_mode="grayscale", 
        class_mode = 'categorical')

    # print(np.shape(training_set.next()[0]))
    # show_pic, yyyyy = training_set.next()
    # img = np.zeros((batch_size, 48, 48))
    # for i in range(batch_size):
    #     img[i, :, :] = show_pic[i,:,:,0]
    # imgs = np.hstack(img)
    # cv2.imshow("train_pic", imgs)
    # cv2.waitKey(0)

    model_info = model.fit(
        training_set, 
        steps_per_epoch=num_train // batch_size, 
        epochs=num_epoch, 
        verbose=1, 
        validation_data = test_set, 
        validation_steps=num_val // batch_size, 
        callbacks=[checkpoint, EarlyStop, adapt_lr, tensorboard_callback])

    # 儲存權重
    model.save_weights('run/train-face/epochs/last_weights.h5')
    model.save('run/train-face/epochs/model_keras.h5')
    print('\n====================================================')
    print('Weights are saved to run/train-face/epochs/last_weights.h5')
    print('Best weights are saved to run/train-face/epochs/best_weights.h5')
    print('====================================================\n')

    # 繪製訓練 & 驗證的準確率值
    model.load_weights('run/train-face/epochs/best_weights.h5')
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
    
    # 結果圖形化
    names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']
    def getLabel(id):
        return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]
    test_set_X, test_set_y = test_set.next()    # tuple to np.array
    # print(type(test_set_X))
    res = np.argmax(model.predict(test_set_X[:9]), axis=-1)
    plt.figure(figsize=(10, 10))

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(test_set_X[i],cmap=plt.get_cmap('gray'))
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=14)
        plt.savefig('run/train-face/epochs/9Pic.png')

    # Confusion Matrix
    results = np.argmax(model.predict(test_set_X), axis=-1)
    cm = confusion_matrix(np.where(test_set_y == 1)[1], results, normalize='true')
    plt.figure(figsize=(9, 9))
    plt.imshow(cm)
    plt.title('Confusion Matrix')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=round(cm[i,j], 2), va='center', ha='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('run/train-face/epochs/Confusion_Matrix.png')

    plt.show()

    

# Predicting mode

elif mode == 'predict':
    # 載入模型
    model.load_weights(weights_dir)

    # 選擇輸入 0: 第一隻攝影機 mp4: Video
    cap = cv2.VideoCapture(source)
    # 調整預設影像大小，預設值很大，很吃效能(只影響串流影像，所以後面還有resize針對影片)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # 儲存影片結果
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('run/results/output_face.mp4',fourcc, 24.0, (640, 360))
    # 編號對應情緒
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # 取得預設的臉部偵測器
    detector = dlib.get_frontal_face_detector()
    # 根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 當攝影機打開時，對每個frame進行偵測
    while(cap.isOpened()):
        # 計算執行時間(起始時間)
        start_time = time.perf_counter()
        # 讀出frame資訊
        ret, frame = cap.read()
        # 改變輸入影像大小
        frame = cv2.resize(frame, (640, 360))
        # print(frame.shape)
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
            # cv2.putText(frame, text, (x2,y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            # 彩色BGR to 灰階
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 給68特徵點辨識取得一個轉換顏色的frame
            landmarks_frame = cv2.cvtColor(frame, cv2. COLOR_BGR2RGB)
            # 找出特徵點位置
            shape = predictor(landmarks_frame, d)
            # 繪製68個特徵點
            # for i in range(68):
                # cv2.circle(frame,(shape.part(i).x,shape.part(i).y), 2,( 0, 0, 255), 1)
            # 裁切圖片(Crop)
            cropped_frame = gray_frame[y1:y2, x1:x2]
            # 升維
            try:
                cropped_frame = np.expand_dims(np.expand_dims(cv2.resize(cropped_frame, (48, 48), cv2.INTER_NEAREST), -1), 0)
            except:
                break
            # 情緒識別
            prediction = model.predict(cropped_frame)
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
        # 如果按下ESC键，就退出
        if cv2.waitKey(10) == 27:
            break
    # 釋放記憶體
    cap.release()
    # 關閉所有視窗
    cv2.destroyAllWindows()


elif mode == 'show_layers':
    from tensorflow.keras.models import Model
    model = load_model('run/train-face/epochs/model_keras.h5')
    model.load_weights(weights_dir)
    model.summary()

    layer0  = Model(inputs=model.input, outputs=model.layers[0].output)
    layer1  = Model(inputs=model.input, outputs=model.layers[1].output)
    layer2  = Model(inputs=model.input, outputs=model.layers[2].output)
    layer3  = Model(inputs=model.input, outputs=model.layers[3].output)
    layer4  = Model(inputs=model.input, outputs=model.layers[4].output)
    layer5  = Model(inputs=model.input, outputs=model.layers[5].output)
    layer6  = Model(inputs=model.input, outputs=model.layers[6].output)
    layer7  = Model(inputs=model.input, outputs=model.layers[7].output)
    layer8  = Model(inputs=model.input, outputs=model.layers[8].output)
    layer9  = Model(inputs=model.input, outputs=model.layers[9].output)
    layer10  = Model(inputs=model.input, outputs=model.layers[10].output)
    layer11  = Model(inputs=model.input, outputs=model.layers[11].output)
    layer12  = Model(inputs=model.input, outputs=model.layers[12].output)
    layer13  = Model(inputs=model.input, outputs=model.layers[13].output)
    layer14  = Model(inputs=model.input, outputs=model.layers[14].output)


    img = cv2.imread('layer_output/test.tiff', cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(np.expand_dims(cv2.resize(img, (48, 48), cv2.INTER_NEAREST), -1), 0)
    feature=layer0.predict(img)
    print(np.shape(feature)) 
    # print(len(feature[0,0,0,:])) 
    for i in range(len(feature[0,0,0,:])):   
        Img_Name = "layer_output/layer0_" + str(i) + ".jpg"
        cv2.imwrite(Img_Name, feature[0, :, :, i])

    feature=layer2.predict(img)
    print(np.shape(feature)) 
    for i in range(len(feature[0,0,0,:])):   
        Img_Name = "layer_output/layer2_" + str(i) + ".jpg"
        cv2.imwrite(Img_Name, feature[0, :, :, i])

    feature=layer6.predict(img)
    print(np.shape(feature)) 
    for i in range(len(feature[0,0,0,:])):   
        Img_Name = "layer_output/layer5_" + str(i) + ".jpg"
        cv2.imwrite(Img_Name, feature[0, :, :, i])

    # feature=layer8.predict(img)
    # print(np.shape(feature)) 
    # for i in range(len(feature[0,0,0,:])):   
    #     Img_Name = "layer_output/layer8_" + str(i) + ".jpg"
    #     cv2.imwrite(Img_Name, feature[0, :, :, i])


    # Confusion Matrix
    from sklearn.metrics import classification_report
    import itertools

    def plot_confusion_matrix_(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.viridis):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="black" if cm[i, j] > thresh else "white")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()


    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(
                            'dataset/test', 
                            target_size = (48, 48), 
                            batch_size = 7178, 
                            color_mode="grayscale", 
                            class_mode = 'categorical')
    test_set_X, test_set_y = test_set.next()    # tuple to np.array

    y_predict = model.predict(test_set_X, batch_size=None, verbose=0, steps=None)

    y_pred = np.argmax(y_predict, axis=-1)
    y_true = np.argmax(test_set_y, axis=-1)
    print(y_pred)
    names = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    print ("7*7 Confusion Matrix")
    print(classification_report(y_true, y_pred, target_names=names))
    print ("**************************************************************")

    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_(cm, classes=names,normalize=True,
                        title="7*7 Confusion Matrix")
    plt.savefig('run/train-face/epochs/Confusion_Matrix0.png')
    plt.show()
    