# 비디오를 원하는 프레임만큼 나누고 이미지로 저장하는 코드

import cv2
import os

path_Video = "/SSD/hackathon/data/data_validation/cap05.mp4"
path_save_image = "/SSD/hackathon/data/video_to_image"
name_save_image = "image_"
name_save_image_format = '.jpg'
count_save_image = 3194
count_current_frame = 0

cap = cv2.VideoCapture(path_Video)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    count_current_frame += 1

    print(count_current_frame)
    
    saveName = name_save_image + str(count_save_image).rjust(6, "0") + name_save_image_format
    save = os.path.join(path_save_image, saveName)
    
    cv2.imwrite(save, frame)
    count_save_image += 1







