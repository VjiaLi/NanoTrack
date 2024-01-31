import os
import cv2
from utils.utils import create_exp

color = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245), (244, 212, 66), (230, 46, 240), (212, 222, 250), (144, 153, 70), (255, 190, 220), (36, 146, 154), (0, 0, 128), (195, 255, 170), (117, 0, 0), (169, 169, 169), (0, 0, 0), (83, 44, 0), (16, 165, 255), (198, 132, 12), (102, 189, 255), (77, 77, 247), (164, 85, 36), (172, 183, 65)]  # BGR
text_color = (255, 255, 255)  # BGR

class Show:
    def __init__(self, args, img):
        self.args = args
        self.img = img
        self.lw = max(round(sum(self.img.shape) / 2 * 0.003), 2)
        self.tf = max(self.lw - 1, 1)
    
    def byte_dets(self, dets, color, im, frame , path):
        if dets.shape[0] != 0:
            xyxys = dets[:, 0:4].astype('int') # float64 to int
            confs = dets[:, 4]
            for xyxy, conf in zip(xyxys, confs):
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                im = cv2.rectangle(
                    self.img,
                    (xyxy[0], xyxy[1]),
                    (xyxy[2], xyxy[3]),
                    color[color],
                    self.lw,
                    cv2.LINE_AA
                )
                w, h = cv2.getTextSize(f'{round(conf,2)}', 0, fontScale=self.lw / 3, thickness=self.tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(im, p1, p2, color[color], -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    im,
                    f'{round(conf,2)}',
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    text_color,
                    self.tf,
                    lineType=cv2.LINE_AA
                )
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(f'{path}\\{frame}.jpg',im)  
    
    def byte_tracks(self, tracks, color, im, frame , path):
        if len(tracks) != 0:
            for track in tracks:
                xyxy = track.xyxy.astype('int') # float64 to int
                conf = track.score
                id = track.track_id
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                im = cv2.rectangle(
                    self.img,
                    (xyxy[0], xyxy[1]),
                    (xyxy[2], xyxy[3]),
                    color[color],
                    self.lw,
                    cv2.LINE_AA
                )
                w, h = cv2.getTextSize(f'id: {id}, conf:{round(conf,2)}', 0, fontScale=self.lw / 3, thickness=self.tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(im, p1, p2, color[color], -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    im,
                    f'id: {id}, conf:{round(conf,2)}',
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    text_color,
                    self.tf,
                    lineType=cv2.LINE_AA
                )
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(f'{path}\\{frame}.jpg',im)  

    def sparse_dets(self, dets, color_id, im, frame, path):
        if len(dets) != 0:
            for det in dets:
                tlbr = det.tlbr.astype('int')
                score = det.score
                p1, p2 = (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3]))
                im = cv2.rectangle(
                    self.img,
                    (tlbr[0], tlbr[1]),
                    (tlbr[2], tlbr[3]),
                    color[color_id],
                    self.lw,
                    cv2.LINE_AA
                )
                w, h = cv2.getTextSize(f'conf:{round(score,2)}', 0, fontScale=self.lw / 3, thickness=self.tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(im, p1, p2, color[color_id], -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    im,
                    f'conf:{round(score,2)}',
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    text_color,
                    self.tf,
                    lineType=cv2.LINE_AA
                )
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(f'{path}\\{frame}.jpg',im)



    def show_dets(self, dets):
        xyxys = dets[:, 0:4].astype('int') # float64 to int
        confs = dets[:, 4]
        clss = dets[:, 5].astype('int') # float64 to int
        if self.args.demo == 'image':
            pass
        elif self.args.demo == 'video':
            if dets.shape[0] != 0:
                for xyxy, conf, cls in zip(xyxys, confs, clss):
                    if cls == 0:
                        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        self.img = cv2.rectangle(
                            self.img,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color[0],
                            self.lw,
                            cv2.LINE_AA
                        )
                        w, h = cv2.getTextSize(f'{round(conf,2)}', 0, fontScale=self.lw / 3, thickness=self.tf)[0]  # text width, height
                        outside = p1[1] - h >= 3
                        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                        cv2.rectangle(self.img, p1, p2, color[0], -1, cv2.LINE_AA)  # filled
                        cv2.putText(
                            self.img,
                            f'{round(conf,2)}',
                            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            text_color,
                            self.tf,
                            lineType=cv2.LINE_AA
                        )

    def show_tracks(self, tracks):
        try:
            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int') # float64 to int
            inds = tracks[:, 7].astype('int') # float64 to int
        except:
            pass
        if self.args.demo == 'image':
            pass
        elif self.args.demo == 'video':
            if tracks.shape[0] != 0:
                for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
                    if cls == 0:
                        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        self.img = cv2.rectangle(
                            self.img,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color[id % len(color)],
                            self.lw,
                            cv2.LINE_AA
                        )
                        w, h = cv2.getTextSize(f'id: {id}, conf:{round(conf,2)}', 0, fontScale=self.lw / 3, thickness=self.tf)[0]  # text width, height
                        outside = p1[1] - h >= 3
                        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                        cv2.rectangle(self.img, p1, p2, color[id % len(color)], -1, cv2.LINE_AA)  # filled
                        cv2.putText(
                            self.img,
                            f'id: {id}, conf:{round(conf,2)}',
                            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            text_color,
                            self.tf,
                            lineType=cv2.LINE_AA
                        )