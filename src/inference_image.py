# 2020년 11월 25일, 이종구
# 학습한 Mask_RCNN 모델 추론 테스트 코드

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_images
from mrcnn import visualize
import balloon
import skimage
import numpy as np

class InferenceConfig(balloon.BalloonConfig):
    # NAME은 학습모델과 동일한 명을 부여
    NAME='airpod'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


##################### Start Code ########################

infer_config = InferenceConfig()
infer_config.display()

# model = modellib.MaskRCNN(mode="inference", model_dir='/SSD/hackathon/Mask_RCNN/', config=infer_config)
dir_name = '/SSD/hackathon_test/snapshots/logs'
# model = modellib.MaskRCNN(mode="inference", model_dir='/SSD/hackathon/Mask_RCNN/samples/balloon/snapshot/', config=infer_config)
model = modellib.MaskRCNN(mode="inference", model_dir=dir_name, config=infer_config)
# callback에 의해 model weights 가 파일로 생성되며, 가장 마지막에 생성된 weights 가 가장 적은 loss를 가지는 것으로 가정. 
weights_path = model.find_last()
print('model path:', weights_path)
# 지정된 weight 파일명으로 모델에 로딩. 
model.load_weights(weights_path, by_name=True)

# Inference를 위해 val Dataset 로딩. 
dataset_val = balloon.BalloonDataset()
dataset_val.load_balloon("/SSD/hackathon_test/data/", "val")
dataset_val.prepare()

print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))


# image_id = np.random.choice(dataset.image_ids) # dataset중에 임의의 파일을 한개 선택. 
image_id = 5
image, image_meta, gt_class_id, gt_bbox, gt_mask=modellib.load_image_gt(dataset_val, infer_config, image_id, use_mini_mask=False)
info = dataset_val.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset_val.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

r = results[0]

# print(results)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
# for i in r['masks']:
    # print(i)
print("shape : ",np.shape(r['masks']))
print("type : ",type(r['masks']))
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
# print(dataset_val.class_names)

# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            # dataset_val.class_names, r['scores'], 
                            # title="Predictions")

# splash = balloon.color_splash(image, r['masks'])
# display_images([splash], cols=1)

def color_splash(image, mask):

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        print(mask)
        # splash = np.where(mask, image, gray).astype(np.uint8)
        image = image[..., ::-1]

    # (1024, 1024, 3)
        temp_image = np.copy(image)
        # for i, value1 in enumerate(temp_image):
            # for j, value2 in enumerate(value1):
                # temp_image[i][j][1] = 200

        temp_image[:,:,1] = 200
        splash = np.where(mask, temp_image, image).astype(np.uint8)
        # splash = np.where(mask, image.100, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


splash = color_splash(image, r['masks'])
display_images([splash], cols=1)

import cv2
# splash = splash[..., ::-1]
cv2.imwrite("test.jpg", splash)

# def apply_mask(image, mask, color, alpha=0.5):
    # """Apply the given mask to the image.
    # """
    # for c in range(3):
        # image[:, :, c] = np.where(mask == 1,
                                  # image[:, :, c] *
                                  # (1 - alpha) + alpha * color[c] * 255,
                                  # image[:, :, c])

    # for c in range(3):
        # print('+++++++++++++++++')
        # masking_list = np.shape(np.where(mask == 1))
        # print(masking_list)
        # image[:, :, c] = np.where(mask == 1,
                                  # (255, 0, 0), (0,0,0))
                                  
    # return image

# N = r['rois'].shape[0]
# colors = visualize.random_colors(N)
# print(np.shape(r['masks']))
# mask = r['masks']
# masked_image = image.astype(np.uint32).copy()

# splash = apply_mask(masked_image, mask, visualize.random_colors(N))

# cv2.imwrite('test.jpg', splash)














