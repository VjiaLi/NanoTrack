# VJia Li 🔥 Nano Tracking

import copy
import warnings

from .coco import CocoDataset
from .xml_dataset import XMLDataset
from .yolo import YoloDataset


def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    if name == "coco":
        warnings.warn(
            "Dataset name coco has been deprecated. Please use CocoDataset instead."
        )
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "yolo":
        return YoloDataset(mode=mode, **dataset_cfg)
    elif name == "xml_dataset":
        warnings.warn(
            "Dataset name xml_dataset has been deprecated. "
            "Please use XMLDataset instead."
        )
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "CocoDataset":
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "YoloDataset":
        return YoloDataset(mode=mode, **dataset_cfg)
    elif name == "XMLDataset":
        return XMLDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")
