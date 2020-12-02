# 2020년 11월 25일, 이종구
# 학습한 Mask_RCNN 모델 동영상 추론 테스트 코드

# from mrcnn.config import Config
# from mrcnn import model as modellib, utils
from mrcnn import model as modellib
# from mrcnn.visualize import display_images
# from mrcnn import visualize
import balloon
import skimage
import numpy as np
import cv2
import time
# import matplotlib.pyplot as plt
# from matplotlib import patches,  lines
# from matplotlib.patches import Polygon
# from skimage.measure import find_contours


def color_splash(image, mask, points, scores):
# def color_splash(image, mask):

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        # print(mask)
        # splash = np.where(mask, image, gray).astype(np.uint8)
        image = image[..., ::-1]

    # (1024, 1024, 3)
        temp_image = np.copy(image)
        # for i, value1 in enumerate(temp_image):
            # for j, value2 in enumerate(value1):
                # temp_image[i][j][1] = 200
                # pass
            # pass
        temp_image[:,:,1] = 200

        splash = np.where(mask, temp_image, image).astype(np.uint8)

        if len(points) != 0:
            for i,j in zip(points, scores):
    
                title = 'cavity : ' + str(j)
                point = (i[0], i[1])
    
                cv2.putText(splash, title, point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        # splash = np.where(mask, image[::[50]], image).astype(np.uint8)
        # splash = np.where(mask, image.100, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_video_color_splash(model, video_input_path=None, video_output_path=None):

    cap = cv2.VideoCapture(video_input_path)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    vid_writer = cv2.VideoWriter(video_output_path, codec, fps, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                 round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 개수: {0:}".format(total))

    frame_index = 0
    success = True
    while True:
        
        hasFrame, image_frame = cap.read()
        if not hasFrame:
            print('End of frame')
            break
        frame_index += 1
        print("frame index:{0:}".format(frame_index), end=" ")
        
        # OpenCV returns images as BGR, convert to RGB
        image_frame = image_frame[..., ::-1]
        start=time.time()
        # Detect objects
        r = model.detect([image_frame], verbose=0)[0]
        print('detected time:', time.time()-start)

        # 실제로 마스킹 되는 부분
        points = r['rois'] 
        print(points)

        scores = r['scores']
        print(scores)

        splash = color_splash(image_frame, r['masks'], points, scores)
        # splash = color_splash(image_frame, r['masks'])
        vid_writer.write(splash)
    
    vid_writer.release()
    cap.release()       
    
    print("Saved to ", video_output_path)


class InferenceConfig(balloon.BalloonConfig):
    # NAME은 학습모델과 동일한 명을 부여
    NAME='cavity'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

##################### Start Code ########################

# infer_config = InferenceConfig()

# dir_name = '/SSD/hackathon/data/snapshots/logs'
# video_input_path = '/SSD/hackathon/data/data_validation/resize_video.mov'
# video_output_path = '/SSD/hackathon/data/data_validation/result_video.mp4'

# model = modellib.MaskRCNN(mode="inference", model_dir=dir_name, config=infer_config)
# # 마지막 스냅샷 지점 불러오기
# weights_path = model.find_last()
# print('model path:', weights_path)
# # 지정된 weight 파일명으로 모델에 로딩. 
# model.load_weights(weights_path, by_name=True)

# test_stamp = time.time()
# detect_video_color_splash(model, video_input_path=video_input_path, video_output_path=video_output_path)
# print("소요시간 : ", time.time() - test_stamp)



