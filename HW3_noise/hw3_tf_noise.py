
from IPython.display import Image
from subprocess import call
import pydotplus
import joblib
from sklearn.tree import export_graphviz
from numpy.core.defchararray import array
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential  # Start NN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense       # Fully-Connected Network
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

from impyute.imputation.cs import mice
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time

from tensorflow.python import keras


# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='predict', help='train/predict')
opt = ap.parse_args()

mode = opt.mode


print('=========================  Part A and D  =========================\n')
# Import the dataset
train_dataset = pd.read_csv('2class_trianing.csv')
test_dataset = pd.read_csv('2class_test.csv')
train_dataset4 = pd.read_csv('4class_trianing.csv')
test_dataset4 = pd.read_csv('4class_test.csv')
# print(train_dataset.columns[1:-1])
# Count the number of NaN
# print(train_dataset)
# print(train_dataset.isna().sum())
# print(test_dataset.isna().sum())
X_train = train_dataset.iloc[:, 1:-1]
y_train = train_dataset.iloc[:, 119]
X_test = test_dataset.iloc[:, 1:-1]
y_test = test_dataset.iloc[:, 119]
X_train4 = train_dataset4.iloc[:, 1:-1]
y_train4 = train_dataset4.iloc[:, 119]
X_test4 = test_dataset4.iloc[:, 1:-1]
y_test4 = test_dataset4.iloc[:, 119]
# 用MICE填補缺失值
X_train = mice(X_train.values)
X_test = mice(X_test.values)
X_train4 = mice(X_train4.values)
X_test4 = mice(X_test4.values)
# print(type(X_train))


def feature_normalize(X):
    # mean of indivdual column, hence axis = 0
    mu = np.mean(X, axis=0)
    # Notice the parameter ddof (Delta Degrees of Freedom)  value is 1
    # Standard deviation (can also use range)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma


# Normalization
X_train, mu, sigma = feature_normalize(X_train)
X_test, mu, sigma = feature_normalize(X_test)
X_train4, mu, sigma = feature_normalize(X_train4)
X_test4, mu, sigma = feature_normalize(X_test4)

# Add noise to train data

X_train2_watts = X_train ** 2
X_train4_watts = X_train4 ** 2
# Set a target SNR
target2_snr_db_train = 10
target4_snr_db_train = 10
# Calculate signal power and convert to dB
X_train2_avg_watts = np.mean(X_train2_watts)
X_train4_avg_watts = np.mean(X_train4_watts)
X_train2_avg_db = 10 * np.log10(X_train2_avg_watts)
X_train4_avg_db = 10 * np.log10(X_train4_avg_watts)
# Calculate noise according to [2] then convert to watts
noise2_avg_db_train = X_train2_avg_db - target2_snr_db_train
noise4_avg_db_train = X_train4_avg_db - target4_snr_db_train
noise2_avg_watts_train = 10 ** (noise2_avg_db_train / 10)
noise4_avg_watts_train = 10 ** (noise4_avg_db_train / 10)
# Generate an sample of white noise
mean_noise = 0
noise2_volts = np.random.normal(
    mean_noise, np.sqrt(noise2_avg_watts_train), X_train2_watts.shape)
noise4_volts = np.random.normal(
    mean_noise, np.sqrt(noise4_avg_watts_train), X_train4_watts.shape)
# Noise up the original signal
X_train2_noise = X_train + noise2_volts
X_train4_noise = X_train4 + noise4_volts


# categorize
y_train4 = to_categorical(y_train4, 4)
y_test4 = to_categorical(y_test4, 4)


if mode == 'train':
    # Import the model for 2 classes
    model = Sequential()

    model.add(Dense(units=200, activation='relu', input_shape=(118,)))
    model.add(Dropout(0.5))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics='accuracy')

    # model for 4 classes
    model4 = Sequential()

    model4.add(Dense(units=200, activation='relu', input_shape=(118,)))
    model4.add(Dropout(0.5))
    model4.add(Dense(units=100, activation='relu'))
    model4.add(Dropout(0.5))
    model4.add(Dense(units=50, activation='relu'))
    model4.add(Dropout(0.5))
    model4.add(Dense(units=32, activation='relu'))
    model4.add(Dropout(0.5))
    model4.add(Dense(units=8, activation='relu'))
    model4.add(Dropout(0.5))
    model4.add(Dense(units=4, activation='softmax'))
    # categorical_crossentropy
    model4.compile(loss='mse',
                   optimizer=optimizer, metrics='accuracy')

    # model with AWGN
    model_noise = Sequential()

    model_noise.add(Dense(units=200, activation='relu', input_shape=(118,)))
    model_noise.add(Dropout(0.5))
    model_noise.add(Dense(units=100, activation='relu'))
    model_noise.add(Dropout(0.5))
    model_noise.add(Dense(units=50, activation='relu'))
    model_noise.add(Dropout(0.5))
    model_noise.add(Dense(units=3, activation='relu'))
    model_noise.add(Dropout(0.3))
    model_noise.add(GaussianNoise(stddev=1))
    model_noise.add(Dense(units=1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model_noise.compile(loss='mse', optimizer=optimizer, metrics='accuracy')

    # model with noise for 4 classes
    model4_noise = Sequential()

    model4_noise.add(Dense(units=200, activation='relu', input_shape=(118,)))
    model4_noise.add(Dropout(0.5))
    model4_noise.add(Dense(units=100, activation='relu'))
    model4_noise.add(Dropout(0.5))
    model4_noise.add(Dense(units=50, activation='relu'))
    model4_noise.add(Dropout(0.5))
    model4_noise.add(Dense(units=32, activation='relu'))
    model4_noise.add(Dropout(0.5))
    model4_noise.add(Dense(units=8, activation='relu'))
    model4_noise.add(Dropout(0.5))
    model4_noise.add(GaussianNoise(stddev=1))
    model4_noise.add(Dense(units=4, activation='softmax'))
    # categorical_crossentropy
    model4_noise.compile(loss='mse',
                         optimizer=optimizer, metrics='accuracy')

    # Train
    # show model info
    model.summary()
    model4.summary()
    checkpoint2 = ModelCheckpoint("DNN_model2_best.h5",
                                  monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='auto',
                                  save_freq='epoch', save_weights_only=False,)
    checkpoint4 = ModelCheckpoint("DNN_model4_best.h5",
                                  monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='auto',
                                  save_freq='epoch', save_weights_only=False,)
    checkpoint2_noise = ModelCheckpoint("DNN_model2_noise_best.h5",
                                        monitor='val_accuracy', verbose=1,
                                        save_best_only=True, mode='auto',
                                        save_freq='epoch', save_weights_only=False,)
    checkpoint4_noise = ModelCheckpoint("DNN_model4_noise_best.h5",
                                        monitor='val_accuracy', verbose=1,
                                        save_best_only=True, mode='auto',
                                        save_freq='epoch', save_weights_only=False,)
    model_info = model.fit(
        X_train,
        y_train,
        epochs=70,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint2]
    )
    model_info4 = model4.fit(
        X_train4,
        y_train4,
        epochs=250,
        verbose=1,
        validation_data=(X_test4, y_test4),
        callbacks=[checkpoint4]
    )
    model_info_noise = model_noise.fit(
        X_train2_noise,
        y_train,
        epochs=70,
        verbose=0,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint2_noise]
    )
    model_info4_noise = model4_noise.fit(
        X_train4_noise,
        y_train4,
        epochs=250,
        verbose=1,
        validation_data=(X_test4, y_test4),
        callbacks=[checkpoint4_noise]
    )

    # Save DNN Weights
    # model.save_weights('last_model_2class.h5')

    # 繪製訓練 & 驗證的準確率值
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.figure()
    plt.subplot(121)
    plt.grid(True, which='both')
    plt.plot(model_info.history['accuracy'])
    plt.plot(model_info.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # 繪製訓練 & 驗證的損失值
    plt.subplot(122)
    plt.grid(True, which='both')
    plt.plot(model_info.history['loss'])
    plt.plot(model_info.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('2Class_History.png')

    # 繪製訓練 & 驗證的準確率值 4 classes
    plt.figure()
    plt.subplot(121)
    plt.grid(True, which='both')
    plt.plot(model_info4.history['accuracy'])
    plt.plot(model_info4.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # 繪製訓練 & 驗證的損失值 4 classes
    plt.subplot(122)
    plt.grid(True, which='both')
    plt.plot(model_info4.history['loss'])
    plt.plot(model_info4.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('4Class_History.png')
    plt.tight_layout()

    # 繪製訓練 & 驗證的準確率值 with noise
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.figure()
    plt.subplot(121)
    plt.grid(True, which='both')
    plt.plot(model_info_noise.history['accuracy'])
    plt.plot(model_info_noise.history['val_accuracy'])
    plt.title('Model accuracy(Noise)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # 繪製訓練 & 驗證的損失值 with noise
    plt.subplot(122)
    plt.grid(True, which='both')
    plt.plot(model_info_noise.history['loss'])
    plt.plot(model_info_noise.history['val_loss'])
    plt.title('Model loss(Noise)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('2Class_History_noise.png')

    # 繪製訓練 & 驗證的準確率值 4 classes with noise
    plt.figure()
    plt.subplot(121)
    plt.grid(True, which='both')
    plt.plot(model_info4_noise.history['accuracy'])
    plt.plot(model_info4_noise.history['val_accuracy'])
    plt.title('Model accuracy(Noise)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # 繪製訓練 & 驗證的損失值 4 classes  with noise
    plt.subplot(122)
    plt.grid(True, which='both')
    plt.plot(model_info4_noise.history['loss'])
    plt.plot(model_info4_noise.history['val_loss'])
    plt.title('Model loss(Noise)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('4Class_History_noise.png')
    plt.tight_layout()
    plt.show()

    # ====================================================================#
    # Random Forest
    # ====================================================================#

    clf = RandomForestClassifier(max_depth=None,
                                 verbose=0,
                                 max_features='sqrt',
                                 random_state=0)
    clf4 = RandomForestClassifier(max_depth=None,
                                  verbose=0,
                                  max_features='sqrt',
                                  random_state=0)
    clf.fit(X_train, y_train)
    clf4.fit(X_train4, y_train4)

    # Save Random forest model
    joblib.dump(clf, "random_forest_2class.joblib")
    joblib.dump(clf4, "random_forest_4class.joblib")

    # Export as dot file
    estimator2 = clf.estimators_[0]
    estimator4 = clf4.estimators_[0]
    tree_dot2 = export_graphviz(estimator2, out_file=None,
                                feature_names=train_dataset.columns[1:-1],
                                class_names=train_dataset.columns[119],
                                rounded=True, proportion=False,
                                precision=2, filled=True)
    tree_dot4 = export_graphviz(estimator4, out_file=None,
                                feature_names=train_dataset4.columns[1:-1],
                                class_names=train_dataset4.columns[119],
                                rounded=True, proportion=False,
                                precision=2, filled=True)

    # Convert to png using system command (requires Graphviz)
    graph2 = pydotplus.graph_from_dot_data(tree_dot2)
    graph4 = pydotplus.graph_from_dot_data(tree_dot4)
    graph2.write_png('2class_tree.png')
    graph4.write_png('4class_tree.png')


# predict
# print(model.predict(X_test))

elif mode == 'predict':
    # evaluate with best model
    model = load_model('DNN_model2_best.h5')
    model4 = load_model('DNN_model4_best.h5')
    model_noise = load_model('DNN_model2_noise_best.h5')
    model4_noise = load_model('DNN_model4_noise_best.h5')

    print('================================================')
    print("Evaluate on test data")
    results2 = model.evaluate(X_test, y_test, verbose=0)
    print("\n2 class: test loss, test acc:", results2)
    results4 = model4.evaluate(X_test4, y_test4, verbose=0)
    print("4 class: test loss, test acc:", results4)
    results2_noise = model_noise.evaluate(X_test, y_test, verbose=0)
    print("2 class(noise): test loss, test acc:", results2_noise)
    results4_noise = model4_noise.evaluate(X_test4, y_test4, verbose=0)
    print("4 class(noise): test loss, test acc:", results4_noise)
    print('================================================')

    clf = joblib.load("random_forest_2class.joblib")
    clf4 = joblib.load("random_forest_4class.joblib")
    print('================================================')
    print('With Random Forest')
    print('2 classes score:', clf.score(X_test, y_test))
    print('4 classes score:', clf4.score(X_test4, y_test4))
    print('================================================')

    # ====================================================================#
    # ====================================================================#
    # ====================================================================#
    print('=========================  Part B  =========================')
    print('=======================  INPUT NOISE  ======================')
    print('===========================  DNN  ==========================\n')

    # evaluate with AWGN
    X_test2_watts = X_test ** 2
    X_test4_watts = X_test4 ** 2
    # Set a target SNR
    target2_snr_db = 10
    target4_snr_db = 10
    # Calculate signal power and convert to dB
    X_test2_avg_watts = np.mean(X_test2_watts)
    X_test4_avg_watts = np.mean(X_test4_watts)
    X_test2_avg_db = 10 * np.log10(X_test2_avg_watts)
    X_test4_avg_db = 10 * np.log10(X_test4_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise2_avg_db = X_test2_avg_db - target2_snr_db
    noise4_avg_db = X_test4_avg_db - target4_snr_db
    noise2_avg_watts = 10 ** (noise2_avg_db / 10)
    noise4_avg_watts = 10 ** (noise4_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise2_volts = np.random.normal(
        mean_noise, np.sqrt(noise2_avg_watts), X_test2_watts.shape)
    noise4_volts = np.random.normal(
        mean_noise, np.sqrt(noise4_avg_watts), X_test4_watts.shape)
    # Noise up the original signal
    X_test2_noise = X_test + noise2_volts
    X_test4_noise = X_test4 + noise4_volts

    print("Evaluate on test data with input + AWGN")
    results2 = model.evaluate(X_test2_noise, y_test, verbose=0)
    print("2 class: test loss, test acc:", results2)
    results4 = model4.evaluate(X_test4_noise, y_test4, verbose=0)
    print("4 class: test loss, test acc:", results4)

    results2_noise = model_noise.evaluate(X_test2_noise, y_test, verbose=0)
    print("2 class(noise): test loss, test acc:", results2_noise)
    results4_noise = model4_noise.evaluate(X_test4_noise, y_test4, verbose=0)
    print("4 class(noise): test loss, test acc:", results4_noise)

    print('=========================  Part B  =========================')
    print('======================  OUTPUT NOISE  ======================')
    print('===========================  DNN  ==========================\n')

    output2 = model.predict(X_test)
    output4 = model4.predict(X_test4)
    output2_noise = model_noise.predict(X_test)
    output4_noise = model4_noise.predict(X_test4)
    # evaluate with AWGN
    output2_watts = output2 ** 2
    output4_watts = output4 ** 2
    output2_watts_noise = output2_noise ** 2
    output4_watts_noise = output4_noise ** 2
    # Set a target SNR
    target2_snr_db_out = 10
    target4_snr_db_out = 10
    target2_snr_db_out_noise = 10
    target4_snr_db_out_noise = 10
    # Calculate signal power and convert to dB
    output2_avg_watts = np.mean(output2_watts)
    output4_avg_watts = np.mean(output4_watts)
    output2_avg_watts_noise = np.mean(output2_watts_noise)
    output4_avg_watts_noise = np.mean(output4_watts_noise)
    output2_avg_db = 10 * np.log10(output2_avg_watts)
    output4_avg_db = 10 * np.log10(output4_avg_watts)
    output2_avg_db_noise = 10 * np.log10(output2_avg_watts_noise)
    output4_avg_db_noise = 10 * np.log10(output4_avg_watts_noise)
    # Calculate noise according to [2] then convert to watts
    noise2_avg_db_out = output2_avg_db - target2_snr_db_out
    noise4_avg_db_out = output4_avg_db - target4_snr_db_out
    noise2_avg_db_out_noise = output2_avg_db_noise - target2_snr_db_out_noise
    noise4_avg_db_out_noise = output4_avg_db_noise - target4_snr_db_out_noise
    noise2_avg_watts_out = 10 ** (noise2_avg_db_out / 10)
    noise4_avg_watts_out = 10 ** (noise4_avg_db_out / 10)
    noise2_avg_watts_out_noise = 10 ** (noise2_avg_db_out_noise / 10)
    noise4_avg_watts_out_noise = 10 ** (noise4_avg_db_out_noise / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise2_volts_out = np.random.normal(
        mean_noise, np.sqrt(noise2_avg_watts_out), output2_watts.shape)
    noise4_volts_out = np.random.normal(
        mean_noise, np.sqrt(noise4_avg_watts_out), output4_watts.shape)
    noise2_volts_out_noise = np.random.normal(
        mean_noise, np.sqrt(noise2_avg_watts_out_noise), output2_watts_noise.shape)
    noise4_volts_out_noise = np.random.normal(
        mean_noise, np.sqrt(noise4_avg_watts_out_noise), output4_watts_noise.shape)
    # Noise up the original signal
    output2_noise = output2 + noise2_volts_out
    output4_noise = output4 + noise4_volts_out
    output2_noise_noise = output2_noise + noise2_volts_out_noise
    output4_noise_noise = output4_noise + noise4_volts_out_noise

    # print(y_test4)
    y_test2_noise = np.zeros((78, 2))
    y_test4_noise = np.zeros((78, 4))
    y_test2_noise_noise = np.zeros((78, 2))
    y_test4_noise_noise = np.zeros((78, 4))
    # print(np.argmax(output4_noise[:, :], axis=1))
    y_test2_noise = np.round(output2_noise, 0)
    y_test4_noise_index = np.argmax(output4_noise[:, :], axis=1)
    y_test4_noise = to_categorical(y_test4_noise_index, 4)
    y_test2_noise_noise = np.round(output2_noise_noise, 0)
    y_test4_noise_index_noise = np.argmax(output4_noise_noise[:, :], axis=1)
    y_test4_noise_noise = to_categorical(y_test4_noise_index_noise, 4)
    # print(y_test4_noise)
    acc_2class = 0.
    acc_4class = 0.
    acc_2class_noise = 0.
    acc_4class_noise = 0.

    for i in range(78):
        if y_test2_noise[i] == y_test[i]:
            acc_2class += 1.
        if y_test2_noise_noise[i] == y_test[i]:
            acc_2class_noise += 1.
        if (y_test4_noise[i, :] == y_test4[i, :]).all():
            acc_4class += 1.
        if (y_test4_noise_noise[i, :] == y_test4[i, :]).all():
            acc_4class_noise += 1.
    acc_2class = acc_2class / 78
    acc_4class = acc_4class / 78
    acc_2class_noise = acc_2class_noise / 78
    acc_4class_noise = acc_4class_noise / 78

    print("Evaluate on test data with output + AWGN")
    print("2 class: test acc:", acc_2class)
    print("4 class: test acc:", acc_4class)
    print("2 class(noise): test acc:", acc_2class_noise)
    print("4 class(noise): test acc:", acc_4class_noise)
