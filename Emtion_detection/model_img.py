# Import Dependencies
from tensorflow.keras.models import Sequential      # 啟動NN
from tensorflow.keras.layers import Conv2D          # Convolution Operation
from tensorflow.keras.layers import MaxPooling2D    # 池化
from tensorflow.keras.layers import Flatten, Dropout        
from tensorflow.keras.layers import Dense           # Fully-Connected Networks
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict',help='train/predict')
ap.add_argument('--weights', default='model.h5', help='model.h5 path')
ap.add_argument('--epochs', type=int, default=30)
opt = ap.parse_args()

# Parameters
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = opt.epochs

mode = opt.mode
weights_dir = opt.weights
print("========================================")
print("Mode =", mode)
print("Weights =", weights_dir)
print("Epochs =", num_epoch)
print("========================================")


# Build and compile model
# initializing CNN
model = Sequential()

model.add(Conv2D(8, kernel_size=(5,5), strides = 1, padding='same', input_shape = (48, 48, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
model.add(Dropout(0.25))
# Second convolutional layer
model.add(Conv2D(16, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
model.add(Dropout(0.25))
# Third convolutional layer
model.add(Conv2D(32, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
model.add(Dropout(0.25))
# Forth convolutional layer
model.add(Conv2D(64, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
model.add(Dropout(0.25))
# fifth convolutional layer
model.add(Conv2D(128, kernel_size=(5,5), strides = 1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding='same'))
model.add(Dropout(0.25))
# sixth convolutional layer
model.add(Conv2D(256, kernel_size=(3,3), strides = 1, padding='same', activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))
# 將 feature maps 攤平放入一個向量中
model.add(Flatten())
# Fully-Connected Networks
model.add(Dense(units = 200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 7, activation = 'softmax'))
# Compiling the CNN
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')


# Training mode

if mode == 'train':
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    model.load_weights('weights/model.h5')  # initial weights
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
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
    model_info = model.fit(
        training_set, 
        steps_per_epoch=num_train // batch_size, 
        epochs=num_epoch, 
        validation_data = test_set, 
        validation_steps=num_val // batch_size)

    model.save_weights('run/train/model.h5')

# Predicting mode
elif mode == 'predict':
    from tensorflow.keras.preprocessing import image
    # 載入模型
    model.load_weights(weights_dir)

    # 計算執行時間(起始時間)
    start_time = time.perf_counter()
    
    # 讀取測試圖片
    test_image = image.load_img('dataset/test/angry/im22.png', color_mode = "grayscale")
    # test_image = image.load_img('KA.AN1.39.tiff', color_mode = "grayscale", target_size=(48,48))
    # test_image = image.load_img('face1.jpg', color_mode = "grayscale", target_size=(48,48))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)
    print('prediction=', prediction)

    # 編號對應情緒
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    print("Test_img has the probability of:")
    print("Angry:     %.4f" % prediction[0][0])
    print("Disgust:   %.4f" % prediction[0][1])
    print("Fearful:   %.4f" % prediction[0][2])
    print("Happy:     %.4f" % prediction[0][3])
    print("Neutral:   %.4f" % prediction[0][4])
    print("Sad:       %.4f" % prediction[0][5])
    print("Surprised: %.4f" % prediction[0][6])

    maxindex = int(np.argmax(prediction))
    print("I guess Test_img is %s" % emotion_dict[maxindex])

    # 計算執行時間(結束時間)
    end_time = time.perf_counter()
    process_time = end_time - start_time
    print('Process time: %.3f' % process_time)



