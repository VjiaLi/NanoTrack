# VJia Li 🔥 Nano Tracking

import numpy as np

from boxmot.motion.kalman_filters.adapters import ByteTrackKalmanFilterAdapter
from boxmot.trackers.nanotrack.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import tlwh2xyah, xywh2tlwh, xywh2xyxy, xyxy2xywh
# from utils.show import Show

class STrack(BaseTrack):
    shared_kalman = ByteTrackKalmanFilterAdapter()

    def __init__(self, det):
        # wait activate
        self.det = det
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.tlwh = xywh2tlwh(self.xywh)  # (xc, yc, w, h) --> (t, l, w, h)
        self.xyah = tlwh2xyah(self.tlwh)
        self.score = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # self.cls = cls

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, a, h)
            ret[2] *= ret[3]  # (xc, yc, a, h)  -->  (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class  NanoTracker(object):
    def __init__(
        self, track_thresh=0.45, match_thresh=0.8, track_buffer=25, frame_rate=30
    ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_buffer = track_buffer
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = ByteTrackKalmanFilterAdapter()

    def split_low_dets(self, low_dets):
        # 创建一个列表用于存储不同遮挡情况的检测框
        non_occluded_dets = []
        occluded_dets_by_other = []
        occluded_dets_of_other = []
        
        # 遍历每一个检测框
        for det in low_dets:
            occluded_by_other = False
            occluded_of_other = False
            x1, y1, x2, y2, conf, _, ind = det

            # 假设当前检测框没有被遮挡
            is_non_occluded = True

            # 遍历其他检测框，判断当前检测框是否被遮挡或遮挡其他检测框
            for other_det in low_dets:
                other_x1, other_y1, other_x2, other_y2, other_conf, _, other_ind = other_det

                if ind == other_ind:
                    continue  # 跳过自身检测框

                # 判断是否有重叠
                overlap = not (x2 < other_x1 or other_x2 < x1 or y2 < other_y1 or other_y2 < y1)

                if overlap:
                    is_non_occluded = False
                    # 判定是被其他框遮挡还是遮挡其他框
                    if conf > other_conf:
                        occluded_by_other = True
                    else:
                        occluded_of_other = True

            # 根据判定结果添加到相应的列表中
            if is_non_occluded:
                non_occluded_dets.append(det)
                continue
            if occluded_of_other:
                occluded_dets_of_other.append(det)
                continue
            if occluded_by_other and not occluded_of_other:
                occluded_dets_by_other.append(det)
            
        return np.array(non_occluded_dets), np.array(occluded_dets_by_other), np.array(occluded_dets_of_other)

    def associate(self, detections, tracks, activated_starcks, refind_stracks, is_fuse, match_thresh):

        dists = iou_distance(tracks, detections)

        if is_fuse:
            dists = fuse_score(dists, detections)
        
        matches, u_track, u_detection = linear_assignment(   # 匈牙利匹配
            dists, thresh=match_thresh
        )

        for itracked, idet in matches:   # 在匹配上的轨迹中寻找是否为丢失的轨迹，如果是的话重新激活，不是的话就更新状态
            track = tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        return matches, u_track, u_detection, activated_starcks, refind_stracks


    def update(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        confs = dets[:, 4]
        """
            visualize
        """
        remain_inds = confs > self.track_thresh   # 取高分检测框

        inds_low = confs > 0.1 
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # 取 0.1 ~ thresh 的检测框，即低分检测框
        dets_second = dets[inds_second]
        non_occluded_dets , occluded_dets_by_other, occluded_dets_of_other = self.split_low_dets(dets_second)
        # print(len(non_occluded_dets)+len(occluded_dets_by_other)+len(occluded_dets_of_other) == len(dets_second))
        dets = dets[remain_inds]
        # show = Show(None, img)
        # show.observe_low_dets(non_occluded_dets, 0, img, self.frame_id)
        # show.observe_low_dets(occluded_dets_of_other, 2, img, self.frame_id)
        # show.observe_low_dets(occluded_dets_by_other, 1, img, self.frame_id)
        if len(dets) > 0:
            """Detections"""
            detections = [STrack(det) for det in dets]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # 将已有的轨迹和丢失的轨迹合并
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        matches, u_track, u_detection, activated_starcks, refind_stracks = self.associate(detections, strack_pool, activated_starcks, refind_stracks, True, self.match_thresh)

        """ Step 3: Second association, with low score detection boxes"""

        # association the untrack to the non_occluded detections while is the low detections

        r_tracked_stracks = [   # 筛选没有匹配上的并且被激活的轨迹
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]

        if len(non_occluded_dets) > 0:
            """Detections"""
            detections_second = [STrack(det) for det in non_occluded_dets]
        else:
            detections_second = []

        matches, u_track, u_detection_second, activated_starcks, refind_stracks = self.associate(detections_second, r_tracked_stracks, activated_starcks, refind_stracks, True, 0.5)

        if len(occluded_dets_by_other) > 0:
            """Detections"""
            for i in u_detection_second:
                occluded_dets_by_other = np.vstack((occluded_dets_by_other, detections_second[i].det))
            detections_third = [STrack(det) for det in occluded_dets_by_other]
        else:
            detections_third = [STrack(detections_second[i].det) for i in u_detection_second]

        rr_tracked_stracks = [ 
            r_tracked_stracks[i]
            for i in u_track
            if r_tracked_stracks[i].state == TrackState.Tracked
        ]

        matches, u_track, u_detection_third, activated_starcks, refind_stracks = self.associate(detections_third, rr_tracked_stracks, activated_starcks, refind_stracks, True, 0.5)
        
        if len(occluded_dets_of_other) > 0:
            """Detections"""
            for i in u_detection_third:
                occluded_dets_of_other = np.vstack((occluded_dets_of_other, detections_third[i].det))
            detections_last = [STrack(det) for det in occluded_dets_of_other]
        else:
            detections_last = [STrack(detections_third[i].det) for i in u_detection_third]

        rrr_tracked_stracks = [ 
            rr_tracked_stracks[i]
            for i in u_track
            if rr_tracked_stracks[i].state == TrackState.Tracked
        ]

        matches, u_track, u_detection_last, activated_starcks, refind_stracks = self.associate(detections_last, rrr_tracked_stracks, activated_starcks, refind_stracks, False, 0.5)

        for it in u_track:
            track = rrr_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.track_id)
            output.append(t.score)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)
        return outputs


# track_id, class_id, conf


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
