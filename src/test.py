# from cv2 import cv2
import cv2

image = cv2.imread('data/val/IMG_1046.JPEG')
temp_image = image.copy()

# color = [x for i, value in enumerate(image)]


for i, value1 in enumerate(image):
    for j, value2 in enumerate(value1):
        # print(value2)
        # print('=======================')
        # [212 230 229]
        # =======================
        # [213 220 229]
        # =======================
        
        if temp_image[i][j][2] >= 200:
            continue
        temp_image[i][j][2] += 50 
        
        # for k, value3 in enumerate(value2):

            # if value3 <= 50:
                # continue
            # if value3 >= 200:
                # continue

            # temp_image[i][j][k] += 50
             
            # value3 += 50
            # pass
        pass
    pass

cv2.imwrite("test_image.jpg", temp_image)










