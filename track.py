import cv2
import numpy as np
import os
import argparse
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from boxmot.utils import ROOT, WEIGHTS, CONFIG
from boxmot.tracker_zoo import create_tracker
from utils.utils import create_exp
from utils.show import Show

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video"
    )
    parser.add_argument('--tracking-method', type=str, default='sparsetrack',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, nanotrack, sparsetrack')
    parser.add_argument("--config", help="model config file path", default= CONFIG / 'nanodet-plus-m_416.yml')
    parser.add_argument("--model", help="model file path", default= WEIGHTS / 'nanodet-plus-m_416.pth')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument("--path", default=r"C:\Users\Lenovo\Desktop\yolo_tracking-master\MOT16_Video\result1.mp4", help="path to images or video")
    parser.add_argument('--conf', type=float, default=0.3,
                        help='confidence threshold')
    parser.add_argument('--only-detect', action='store_true',
                        help='only display detection')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs' / 'test',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    args = parser.parse_args()
    return args

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
        args.device,
        args.half,
        args.per_class
    )

    if hasattr(tracker, 'model'):
            tracker.model.warmup()
    
    return tracker

class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

def main():
    args = parse_args()
    local_rank = 0

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    logger.log('Press "Esc", "q" or "Q" to exit.')
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    tracker = on_predict_start(args)
    
    if args.save:
        out = create_exp(args)

    if args.demo == "image":
        pass
    elif args.demo == "video" or args.demo == "webcam":
        vid = cv2.VideoCapture(args.path)
        while True:
            ret, im = vid.read()
            if ret:
                show = Show(args, im)
                meta, res = predictor.inference(im)
                all_box = []
                for label in res[0]:
                    for bbox in res[0][label]:
                        score = bbox[-1]
                        if score > args.conf and label == 0:
                            x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                            all_box.append([x0, y0, x1, y1, score, label])

                dets = np.array(all_box)
                if dets.shape[0] == 0:
                    continue
                if args.only_detect:
                    show.show_dets(dets)
                else:
                    tracks = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls, ind) 
                    show.show_tracks(tracks)
                    
                if args.show:
                    # show image with bboxes, ids, classes and confidences
                    cv2.imshow('frame', im)

                if args.save:
                    out.write(im)
                    
                    # break on pressing q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        vid.release()
        cv2.destroyAllWindows()

    if args.save:
        out.release()

if __name__ == '__main__':
    main()
