import inference_video as inf
from mrcnn import model as modellib
import time

def run():
    infer_config = inf.InferenceConfig()

    dir_name = '/SSD/hackathon/data/snapshots/logs'
    video_input_path = '/SSD/hackathon/data/data_validation/validation_video.mp4'
    video_output_path = '/SSD/hackathon/src/static/video/outputVideo.mp4'

    model = modellib.MaskRCNN(mode="inference", model_dir=dir_name, config=infer_config)
    # 마지막 스냅샷 지점 불러오기
    weights_path = model.find_last()
    print('model path:', weights_path)
    # 지정된 weight 파일명으로 모델에 로딩. 
    model.load_weights(weights_path, by_name=True)

    test_stamp = time.time()
    inf.detect_video_color_splash(model, video_input_path=video_input_path, video_output_path=video_output_path)
    print("소요시간 : ", time.time() - test_stamp)

    return 0



