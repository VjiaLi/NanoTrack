import time
import torch
import argparse
import cv2
import numpy as np
from torch.backends import cudnn
from matplotlib import colors
from pathlib import Path
from utils.show import Show
from boxmot.utils import ROOT, WEIGHTS, CONFIG, VIDEO
from boxmot.tracker_zoo import create_tracker
from utils.utils import write_mot_results
from efficientdet.backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video, get_image_list, aspectaware_resize_padding, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video"
    )
    parser.add_argument('--tracking-method', type=str, default='bytetrack',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, nanotrack, sparsetrack')
    parser.add_argument("--model", help="model file path", default= WEIGHTS / 'efficientdet-d0.pth')
    parser.add_argument('--reid-model', type=Path, default= WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument("--path", default= VIDEO / 'mot04.mp4', help="path to images or video")
    parser.add_argument('--conf', type=float, default=0.2,
                        help='confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.3,
                        help='nms threshold')
    parser.add_argument('--classes', nargs='+', type=str, default=['0'],
                    help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--only-detect', action='store_true',
                        help='only display detection')
    parser.add_argument('--use-cuda', default='True',
                        help='whether use cuda')
    parser.add_argument('--project', default=ROOT / 'runs' / 'test',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save-mot', action='store_true',
                    help='...')
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
        'cuda:0',
        args.half,
        args.per_class
    )

    if hasattr(tracker, 'model'):
            tracker.model.warmup()
    
    return tracker

def main():
    args = parse_args()
    compound_coef = 0
    force_input_size = None  # set None to use default size

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    tracker = on_predict_start(args)

    if args.demo == "image":
        img_path = get_image_list(args.path)
        for frame_idx, image_name in enumerate(img_path):
            ori_img = cv2.imread(image_name)
            normalized_img = (ori_img[..., ::-1] / 255 - mean) / std
            img_meta = aspectaware_resize_padding(normalized_img, input_size, input_size, means=None)
            framed_img = img_meta[0]
            framed_meta = img_meta[1:]
            if args.use_cuda:
                x = torch.stack([torch.from_numpy(framed_img).cuda()], 0)
            else:
                x = torch.stack([torch.from_numpy(framed_img)], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                ratios=anchor_ratios, scales=anchor_scales)
            model.load_state_dict(torch.load(str(args.model), map_location='cpu'))
            model.requires_grad_(False)
            model.eval()
            if args.use_cuda:
                model = model.cuda()
            if use_float16:
                model = model.half()
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                args.conf, args.nms_threshold)
                
            # out = invert_affine(framed_meta, out)
            
            if framed_meta is float:
                out[0]['rois'][:, [0, 2]] = out[0]['rois'][:, [0, 2]] / framed_meta
                out[0]['rois'][:, [1, 3]] = out[0]['rois'][:, [1, 3]] / framed_meta
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = framed_meta
                out[0]['rois'][:, [0, 2]] = out[0]['rois'][:, [0, 2]] / (new_w / old_w)
                out[0]['rois'][:, [1, 3]] = out[0]['rois'][:, [1, 3]] / (new_h / old_h)

            if len(out[0]['rois']) == 0:
                continue

            ori_img = ori_img.copy()
            all_box = []
            print(out)
            for j in range(len(out[0]['rois'])):
                x1, y1, x2, y2 = out[0]['rois'][j].astype(int)
                obj = obj_list[out[0]['class_ids'][j]]
                score = float(out[0]['scores'][j])
                if obj == 'person':
                    all_box.append([x1, y1, x2, y2, score, 0])

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
    elif args.demo == "video" or args.demo == "webcam":
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
        model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
        model.requires_grad_(False)
        model.eval()
        if args.use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()
        # Box
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        # Video capture
        cap = cv2.VideoCapture(str(args.path))

        # Output video settings
        output_path = 'videos/output.avi'
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_res = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            show = Show(args, frame)
            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)
            
            if args.use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            # model predict
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                args.conf, args.nms_threshold)

            # result
            out = invert_affine(framed_metas, out)

            for i in range(len(ori_imgs)):
                if len(out[i]['rois']) == 0:
                    continue
                all_box = []
                for j in range(len(out[i]['rois'])):
                    x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
                    obj = obj_list[out[i]['class_ids'][j]]
                    score = float(out[i]['scores'][j])
                    if obj == 'person':
                        all_box.append([x1, y1, x2, y2, score, 0])

                dets = np.array(all_box)

                if dets.shape[0] == 0:
                    continue

                tracks = tracker.update(dets, ori_imgs[i]) # --> (x, y, x, y, id, conf, cls, ind) 

                show.show_tracks(tracks)

                if args.show:
                    # show image with bboxes, ids, classes and confidences
                    cv2.imshow('frame', ori_imgs[i])

                if args.save:
                    out_res.write(ori_imgs[i])

                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

if __name__ == '__main__':
    main()