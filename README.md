# NanoTrack

## Abstract
We introduce NanoTrack, a novel multi-object tracking (MOT) method that leverages light-weight object detectors to enhance tracking performance in real-time applications where computational resources are scarce. While light-weight detectors are efficient, they often produce an imbalance in detection quality, generating a significant number of low-scoring detections that pose challenges for tracking algorithms. Our approach innovatively utilizes these low-scoring detections for track initialization and maintenance, addressing the shortcomings observed in existing tracking by two-stage tracking methods like ByteTrack, which struggle with the abundance of low-scoring detections. By integrating two new light-weight modules, Refind High Detection (RHD) and Duplicate Track Checking (DTC), NanoTrack effectively incorporates low-scoring detections into the tracking process. Additionally, we enhance the pseudo-depth estimation technique for improved handling in dense target environments, mitigating issues like ID Switching. Our comprehensive experiments demonstrate that NanoTrack surpasses state-of-the-art two-stage TBD methods, including ByteTrack and SparseTrack, on benchmark datasets such as MOT16, MOT17, and MOT20, thereby establishing a new standard for MOT performance using light-weight detectors.

![overview](asserts/overview.png)