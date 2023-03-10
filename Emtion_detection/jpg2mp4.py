import cv2
import os

image_folder = 'jaffedbase'
video_name = 'jaffedbase/VIDEO/video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".tiff")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 4.0, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()