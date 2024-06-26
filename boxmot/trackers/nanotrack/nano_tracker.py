import numpy as np
# from tracker import pbcvt 
from .kalman_filter import KalmanFilter
from .matching import *
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, det):

        self.score = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        
        # wait activate
        self._tlwh = np.asarray(self.tlbr_to_tlwh(det[0:4]), dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.deep_vector = self._get_deep_vec()
        self.ini = False

        self.tracklet_len = 0
        
    def _get_deep_vec(self):
        cx = self._tlwh[0] + 0.5 * self._tlwh[2]
        y2 = self._tlwh[1] +  self._tlwh[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float)
    
    def speed_direction(self, next):
        cx1, cy1 = (self.tlbr[0] + self.tlbr[2]) / 2.0, (self.tlbr[1] + self.tlbr[3]) / 2.0
        cx2, cy2 = (next.tlbr[0] + next.tlbr[2]) / 2.0, (next.tlbr[1] + next.tlbr[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1])
        norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
        return speed / norm

    def speed(self, next):
        cx1, cy1 = (self.tlbr[0] + self.tlbr[2]) / 2.0, (self.tlbr[1] + self.tlbr[3]) / 2.0
        cx2, cy2 = (next.tlbr[0] + next.tlbr[2]) / 2.0, (next.tlbr[1] + next.tlbr[3]) / 2.0
        speed = np.array([cy2 - cy1, cx2 - cx1]) + 1e-3
        return speed
    
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
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
        self.pre_pos = self.tlwh
        self.v = self.speed(new_track)

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.det_ind = new_track.det_ind

        self.score = new_track.score

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)# np.kron 
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov
    
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret
    @property
    # @jit(nopython=True)
    def deep_vec(self):
        """Convert bounding box to format `((top left, bottom right)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        y2 = ret[1] +  ret[3]
        lendth = 2000 - y2
        return np.asarray([cx, y2, lendth], dtype=np.float)

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret
    
    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)
    
    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class BoundingBoxProcessor:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank for balancing the tree
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                self.rank[root_y] += 1

    def preprocess(self, objs):
        self.parent = {i: i for i in range(len(objs))}
        self.rank = {i: 0 for i in range(len(objs))}
        res_objs = []

        for i, obj in enumerate(objs):
            x1, y1, x2, y2 = obj.tlbr

            for j, other_obj in enumerate(objs[i + 1:]):
                other_x1, other_y1, other_x2, other_y2 = other_obj.tlbr
                overlap = not (x2 < other_x1 or other_x2 < x1 or y2 < other_y1 or other_y2 < y1)

                if overlap:
                    self.union(i, i + 1 + j)

        components = {}
        for i in range(len(objs)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(objs[i])

        res_objs = list(components.values())
        return res_objs

class NanoTracker(object):
    def __init__(self, track_thresh = 0.45, match_thresh = 0.8, confirm_thresh = 0.7,track_buffer = 25, down_scale = 4 , depth_levels = 1, depth_levels_low = 3, frame_rate = 30):
        self.tracked_stracks = []  # type: list[STrack] 
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.pre_high = []
        self.pre_low = []
        self.frame_id = 0
 
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh
        self.match_thresh = match_thresh
        self.confirm_thresh = confirm_thresh
        self.pre_img = None
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.preprocessor = BoundingBoxProcessor()
        self.down_scale = down_scale
        self.layers = depth_levels 
        self.depth_levels_low = depth_levels_low
    
    def get_deep_range(self, objs, step, objs_ori):
        final_mask = [np.array([False] * len(objs_ori)) for _ in range(step)]
        #print(final_mask)
        for obj in objs:
            col = []
            for t in obj:
                lend = (t.deep_vec)[2]
                col.append(lend)
            max_len, mix_len = max(col), min(col)
            if max_len != mix_len:
                deep_range =np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
                if deep_range[-1] < max_len:
                    deep_range = np.concatenate([deep_range, np.array([max_len],)])
                    deep_range[0] = np.floor(deep_range[0])
                    deep_range[-1] = np.ceil(deep_range[-1])
            else:    
                deep_range = [mix_len,] 
            masks = self.get_sub_mask(deep_range, col)     

            for j, mask in enumerate(masks):
                idx = np.argwhere(mask == True)
                for idd in idx:
                    det = obj[idd[0]]
                    index = objs_ori.index(det)
                    final_mask[j][index] = True
        #print(final_mask)
        return final_mask
    
    def get_sub_mask(self, deep_range, col):
        mix_len=deep_range[0]
        max_len=deep_range[-1]
        if max_len == mix_len:
            lc = mix_len   
        mask = []
        for d in deep_range:
            if d > deep_range[0] and d < deep_range[-1]:
                mask.append((col >= lc) & (col < d)) 
                lc = d
            elif d == deep_range[-1]:
                mask.append((col >= lc) & (col <= d)) 
                lc = d 
            else:
                lc = d
                continue
        return mask
    
    def IDCM(self, detections, tracks, activated_starcks, refind_stracks, levels, thresh, curr_img, stage, frame, is_fuse):
        if len(detections) > 0:
            detections_pro = self.preprocessor.preprocess(detections)
            det_mask = self.get_deep_range(detections_pro, levels, detections) 
        else:
            det_mask = []

        if len(tracks)!=0:
            tracks_pro = self.preprocessor.preprocess(tracks)
            track_mask = self.get_deep_range(tracks_pro, levels, tracks)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if  len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])

            for i, (dm, tm) in enumerate(zip(det_mask, track_mask)):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)
                
                # search det 
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks
                
                dists = iou_distance(track_, det_)
                if is_fuse:
                    dists = fuse_score(dists, det_)
                matches, u_track_, u_det_ = linear_assignment(dists, thresh)
                for itracked, idet in matches:
                    track = track_[itracked]
                    det = det_[idet]

                    if track.state == TrackState.Tracked:
                        track.update(det_[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                u_tracks = [track_[t] for t in u_track_]
                u_detection = [det_[t] for t in u_det_]

            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections
            
        return activated_starcks, refind_stracks, u_tracks, u_detection   
        
    def update(self, dets, curr_img = None):

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        self.frame_id += 1
        if self.frame_id == 1:
            self.pre_img = None
            
        # init stracks
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        # current detections
        scores = dets[:, 4]

        # divide high-score dets and low-scores dets
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        # tracks preprocess
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        
        # init high-score dets
        if len(dets) > 0:
            detections = [STrack(det) for det in dets]   
        else:
            detections = []

        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(det) for det in dets_second]   
        else:
            detections_second = []

        dists = iou_distance(self.pre_high, detections_second)

        true_indices = [index for index, value in enumerate(np.any(dists < 0.2, axis=0)) if value]

        selected_detections = [detections_second[index] for index in true_indices]
        detections.extend(selected_detections)

        detections_second = [det for index, det in enumerate(detections_second) if index not in true_indices]

        dists = iou_distance(self.pre_low, detections_second)
        true_indices = [index for index, value in enumerate(np.any(dists < 0.01, axis=0)) if value]
        selected_detections = [detections_second[index] for index in true_indices]
        for selected_detection in selected_detections:
            selected_detection.ini = True
        self.pre_high = detections
        self.pre_low = detections_second

        # get strack_pool   
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # predict the current location with KF
        STrack.multi_predict(strack_pool)
        """
        show3 = Show(args, curr_img)
        show3.sparse_tracks(tracked_stracks, 2,curr_img, self.frame_id, str(RESULT/ 'sparse_det_high_04'))
        """

        # use GMC: for mot20 dancetrack--unenabled GMC: 368 - 373
        """
        if self.pre_img is not None:
            warp = pbcvt.GMC(curr_img, self.pre_img, self.down_scale)
            pass
        else:
            warp = np.eye(3,3)
            pass
        STrack.multi_gmc(strack_pool, warp[:2, :])
        STrack.multi_gmc(unconfirmed, warp[:2, :])
        """

        # IDCM
        activated_starcks, refind_stracks, u_track, u_detection_high = self.IDCM(
                                                                                detections, 
                                                                                strack_pool, 
                                                                                activated_starcks,
                                                                                refind_stracks, 
                                                                                self.layers, 
                                                                                self.match_thresh,
                                                                                curr_img,
                                                                                1,
                                                                                self.frame_id, 
                                                                                is_fuse=True)  

        # association the untrack to the low score detections
        r_tracked_stracks = [t for t in u_track if t.state == TrackState.Tracked]  

        # IDCM
        activated_starcks, refind_stracks, u_strack, u_detection_sec = self.IDCM(
                                                                                detections_second, 
                                                                                r_tracked_stracks, 
                                                                                activated_starcks, 
                                                                                refind_stracks, 
                                                                                self.depth_levels_low, 
                                                                                0.4, 
                                                                                curr_img,
                                                                                2,
                                                                                self.frame_id,
                                                                                is_fuse=False,
                                                                                )
        
        for track in u_strack:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        for det in u_detection_sec:
            if det.ini is True:
                u_detection_high.append(det)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame 
        detections = [d for d in u_detection_high]
        dists = iou_distance(unconfirmed, detections)
        #if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh = self.confirm_thresh) 
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        oai_track = joint_stracks(self.tracked_stracks, self.lost_stracks)

        u_detections = [detections[idet] for idet in u_detection]
        dists = iou_distance(oai_track, u_detections)
        true_indices = [index for index, value in enumerate(np.any(dists < 0.7, axis=0)) if value]
        u_detection = [det for index, det in enumerate(u_detections) if index not in true_indices]

        """ Step 4: Init new stracks"""
        for track in u_detection:

            """
            if track.score < self.det_thresh:
                continue
            """

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.tlbr)
            output.append(t.track_id)
            output.append(t.score)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)
        self.pre_img = curr_img
        return outputs

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