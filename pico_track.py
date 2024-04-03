import numpy as np
import cv2

from pathlib import Path
from picodet.utils.time_utils import TimeUtils
from picodet.utils.detection_infer_utils import DetectionInferUtils
from picodet.utils.file_utils import FileUtils
from picodet.process.infer.detection_predict import main as main_detection
from picodet.utils.constant import Constants
from boxmot.utils import  CONFIG, OUTPUT, ROOT
from utils.utils import get_image_list
from picodet.core.workspace import load_config, merge_config
from picodet.utils.detection_cli_utils import merge_args
from picodet.utils.detection_common_utils import DetectionCommonUtils
from picodet.engine.detection_trainer import DetectionTrainer
from utils.utils import write_mot_results
from boxmot.tracker_zoo import create_tracker

base_dir_image = f"{Constants.DATA_DIR}/ocr/imgs_words/ch"
base_dir = Constants.WORK_DIR
run_time = TimeUtils.now_str_short()
detection_config_dir = f"{Constants.WORK_DIR}/configs"
checkpoint_base_url = "https://paddledet.bj.bcebos.com/models"
checkpoint_base_url_paddledet = "https://bj.bcebos.com/v1/paddledet/models"
checkpoint_base_url_ppstructure = "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/"
checkpoint_base_url_pedestrian = "https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_enhance/"

def get_model_class(config_file):
        begin_index = len(f"{detection_config_dir}/")
        config_name = config_file[begin_index:]
        end_index = config_name.find("/")
        model_class = config_name[:end_index]
        return model_class

def on_predict_start(args):
    """
    Initialize tracker for object tracking during prediction
    """

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (args.tracking_method + '.yaml')

    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        args.reid_model,
        'cuda:0',
        args.half,
        args.per_class
    )

    if hasattr(tracker, 'model'):
            tracker.model.warmup()
    
    return tracker

def main():
    args = DetectionInferUtils.init_args()
    config_name = args.config
    config_name = config_name if not config_name.endswith(".yml") else config_name[:-4]

    config_file = f"{CONFIG}/picodet/{config_name}.yml"

    model_name = FileUtils.get_file_name(config_file)
    layout_zh = False

    predict_labels = None
    if model_name in ["picodet_lcnet_x1_0_layout", ]:
        layout_zh = True
        base_url = checkpoint_base_url_ppstructure
        predict_labels = "~/PaddleOCR/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt"
    elif model_name in ["picodet_s_192_pedestrian", "picodet_s_320_pedestrian",
                        "picodet_s_192_lcnet_pedestrian", "picodet_s_320_lcnet_pedestrian"]:
        base_url = checkpoint_base_url_pedestrian
        # model_name = str(model_name).replace("_pedestrian", "_pedestrian")
    else:
        base_url = checkpoint_base_url

    if layout_zh:
        checkpoint_file = f"{base_url}/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams"
    else:
        checkpoint_file = f"{base_url}/{model_name}.pdparams"

    args.config = config_file
    args.opt = {
        "use_gpu": True,
        "weights": checkpoint_file
    }

    if layout_zh:
        args.opt["num_classes"] = 10

    img_path = get_image_list(args.path)
    args.predict_labels = predict_labels
    model_class = get_model_class(config_file)
    args.output_dir = f"{OUTPUT}/detection/{model_class}/{model_name}/inference_results/{run_time}"
    args.do_transform = True
    cfg = load_config(args.config)
    merge_args(cfg, args)
    merge_config(args.opt)
    DetectionCommonUtils.check_config(cfg)
    # build trainer
    trainer = DetectionTrainer(cfg, mode='test')
    # load weights
    weight_output_dir = f"{args.output_dir}/network/{FileUtils.get_file_name(args.config)}"
    trainer.load_weights(cfg.weights, do_transform=args.do_transform, output_dir=weight_output_dir)
    tracker = on_predict_start(args)

    for frame_idx, image_name in enumerate(img_path):
        ori_img = cv2.imread(image_name)
        # get inference images
        images = DetectionInferUtils.get_test_images(args.infer_dir, image_name)
        # inference
        results = trainer.predict(
                    images,
                    draw_threshold=args.draw_threshold,
                    output_dir=args.output_dir,
                    save_results=args.save_results,
                    visualize=args.visualize)
        
        all_box = []

        for j in range(len(results[0]['bbox'])):
            x1, y1, x2, y2 = results[0]['bbox'][j][2:6].astype(int)
            score = float(results[0]['bbox'][j][1])
            cls = results[0]['bbox'][j][0].astype(int)
            if cls == 0 and score > args.conf:
                all_box.append([x1, y1, x2, y2, score, cls])
       
        dets = np.array(all_box)

        if dets.shape[0] == 0:
            continue

        tracks = tracker.update(dets, ori_img) # --> (x, y, x, y, id, conf, cls, ind) 
        
        mot_txt_path = Path(args.project) / Path(args.name) / 'mot' / (Path(args.path).parent.name + '.txt')

        if args.save_mot:
            write_mot_results(
                mot_txt_path,
                tracks,
                frame_idx,
            )

if __name__ == '__main__':
    main()
