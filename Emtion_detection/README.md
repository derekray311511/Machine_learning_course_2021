# **ALL this code is written by myself**
# **With the use of open source library**
# Important Library: 
# Tensorflow(Keras), matplotlib, OpenCV, Dlib, keras-video-generators 


If you want to get face landmark with opencv, then run (Pic)
# python face_detect_img.py

If you want to get face landmark with dlib, then run (Pic)
# python face_landmark_img.py

If you want to get face landmark with dlib, then run (Video)
# python face_landmark.py

To recognize emotion on video, then run (Video)(CNN)
# python model.py 

To recognize emotion on video, then run (Video)(Model for JAFFE)(CNN)
# python model_JAFFE.py 

To recognize emotion on video, then run (Video)(Model for facial landmark point)
# python model_point.py 


Picture output will be result.jpg
Video output will be output_face.mp4

# version v1: Detect face_landmark
# version v2: Add model of emotion recognition (model_img.py)
# version v3: Stream emotion recognition(model.py)
    face detect + landmark ~= 36 ms
    emotion recognition ~= 48 ms *for each face
# version v3.1: Add checkpoint(best_model.h5)
    improve training experience
    show model summary when training
# version v4: Seperate model to 3 kind of model
    model.py
    model_point.py
    model_JAFFE.py
# version v5: Visualization improvement/ Add RNN model
    RNN_model.py 
    Video_dataset(face)(GIF dataset)
    Get_face.py (get face part in GIF)
    jpg2mp4.py (transform GIF pictures to video dataset)

對於這次實驗的結果，我想講一下模型選擇的過程。
我們使用smellgood.mp4這部影片當作是評估我們模型泛化能力的一個標準，
實際測試下發現VGG-RNN，也就是訓練完CNN準確度比較高的MODEL對於影片的辨識結果反而較差。
相比之下，Small-CNN-RNN，也就是較簡單的MODEL辨識效果反而好很多。
這也是為什麼當初我要選擇使用較小的這個MODEL來訓練，因為我希望可以避免過度擬合。
不只是使用Data Augmentation，這只是影像資料前處理的基本。
因為如果模型過於複雜，盲目的選擇複雜的模型來追求accuracy，
可能反而會導致模型對於這份Dataset過度擬合。
我們的實驗結果很好的說明了這個現象。

至於RNN訓練的部分，因為Dataset實在是不夠大，所以品質不太好。
不管是VGG-RNN還是Small-CNN-RNN，在訓練的過程中震盪都非常大，
而且每次訓練的accuracy都有蠻大的浮動，這就是因為Dataset不夠大所導致的結果。
雖然有利用VideoDataAugmentation來增加影片的資料量了，但還是無法避免資料過小的問題。
因為學會使用RNN花了我們很多時間，所以最後能夠蒐集機料的時間就被壓縮了。
要改善RNN的訓練結果我想最好的方法就是多花時間蒐集影片資料，
並且嚴謹的選擇資料。