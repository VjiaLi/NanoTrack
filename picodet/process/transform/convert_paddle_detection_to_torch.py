


# @File    ：convert_paddle_detection_to_torch.py

# @Date    ：2022/11/10 14:12
import re
from collections import OrderedDict

import os
import torch

from picodet.process.transform.convert_paddle_config import MODEL_PARAMS_CONFIG
from picodet.utils.match_utils import MatchUtils
from picodet.process.transform.convert_paddle_to_torch_base import ConvertPaddleToTorchBase
from picodet.utils.logger_utils import logger
from picodet.utils.file_utils import FileUtils

"""
转换  paddle detection model to torch
"""


class ConvertPaddleDetectionModelToTorch(ConvertPaddleToTorchBase):

    def __init__(self):
        super().__init__()

        self.number_replace_name_list = []

    def get_model_class(self, model_name):
        """
        获取模型名称

        :param model_name:
        :return:
        """
        model_class = ""
        if model_name is not None:
            end_index = model_name.find("_")
            if end_index > -1:
                model_class = model_name[:end_index]
            else:
                model_class = model_name

        return model_class

    def get_model_class_pattern(self, model_name):
        """
        获取不同模型的 rename list

        :param model_name:
        :return:
        """
        pattern = self.get_model_match_pattern(model_name=model_name, name="rename")
        return pattern

    def get_model_weight_pattern(self, model_name):
        """
        获取不同模型的 transpose list

        :param model_name:
        :return:
        """
        pattern = self.get_model_match_pattern(model_name=model_name, name="transpose", default="LM")
        return pattern

    def get_model_filter_pattern(self, model_name):
        """
        获取不同模型的 filter list

        :param model_name:
        :return:
        """
        pattern = self.get_model_match_pattern(model_name=model_name, name="filter")
        return pattern

    def get_model_add_pattern(self, model_name):
        """
        获取不同模型的 add list

        :param model_name:
        :return:
        """
        pattern = self.get_model_match_pattern(model_name=model_name, name="add")
        return pattern

    def get_model_prefix_all_pattern(self, model_name):
        """
        获取不同模型的 prefix_all

        :param model_name:
        :return:
        """
        pattern = self.get_model_match_pattern(model_name=model_name, name="prefix_all")

        if isinstance(pattern, list) and len(pattern) > 0:
            pattern = pattern[0]
        else:
            pattern = ""
        logger.info(f"prefix_all: {model_name} -> {pattern}")

        return pattern

    def get_model_prefix_pattern(self, model_name, name, default=None):
        """
        获取不同模型的 prefix_all

        :param model_name:
        :param name:
        :param default:
        :return:
        """
        pattern = self.get_model_match_pattern(model_name=model_name, name=name, default=None)

        if isinstance(pattern, list) and len(pattern) > 0:
            pattern = pattern
        else:
            pattern = ""
        logger.info(f"{name}: {model_name} -> {pattern}")

        return pattern

    def get_model_match_pattern(self, model_name, name, default=None):
        """
        获取不同的参数配置

        :param model_name:
        :param name:
        :param default:
        :return:
        """
        model_class = self.get_model_class(model_name)

        # if model_class in ["yolov7p6"]:
        #     model_class = "yolov7"

        pattern = []
        if model_class in MODEL_PARAMS_CONFIG:
            model_config = MODEL_PARAMS_CONFIG.get(model_class)
            pattern = model_config.get(name, [])

        if len(pattern) == 0 and default is not None:
            pattern = MODEL_PARAMS_CONFIG[default][name]

        return pattern

    def number_replace(self, key):
        """
             "0.0.":  -> "0_0."

        :param key:
        :return:
        """
        new_key = key
        if key is not None and len(key) > 0:
            res = MatchUtils.match_pattern_extract(key, pattern=MatchUtils.PATTERN_PADDLE_NUMBER_REPLACE)
            if len(res) > 0:
                key1 = res[0][0]
                key2 = res[0][1]
                key3 = res[0][2]
                key4 = res[0][3]
                new_key = f"{key1}{key2}_{key3}.{key4}"

        return new_key

    def check_need_number_replace(self, model_name, key=None):
        """
        检测是否需要转换

        :param model_name:
        :return:
        """
        pattern = [
            # re.compile(r'picodet_[sml]_(320|416|640)_coco')
            re.compile(r'head.conv_feat.cls_conv_(dw|pw)')
        ]
        match_flag = MatchUtils.match_pattern_list_flag(texts=key, pattern_list=pattern)
        return match_flag

    def check_need_csp_darknet_replace(self, model_name, key=None):
        """
        检测是否需要转换  layers{}_stage{}_conv_layer

            layers1.stage1.conv_layer -> layers1_stage1_conv_layer
        :param model_name:
        :return:
        """
        pattern = [
            re.compile(r'layers(\d+)\.stage(\d+)\.(conv|csp|spp|sppf)_layer')
        ]
        match_flag = MatchUtils.match_pattern_list_flag(texts=key, pattern_list=pattern)
        return match_flag

    def csp_darknet_replace(self, key):
        """
            layers1.stage1.conv_layer -> layers1_stage1_conv_layer

        :param key:
        :return:
        """
        new_key = key
        if key is not None and len(key) > 0:
            # layers(\d+)\.stage(\d+)\.conv_layer
            res = MatchUtils.match_pattern_extract(key, pattern=MatchUtils.PATTERN_PADDLE_CSP_DARKNET_REPLACE)
            if len(res) > 0:
                key1 = res[0][0]
                key2 = res[0][1]
                key3 = res[0][2]
                key4 = res[0][3]
                new_key = f"{key1}layers{key2}_stage{key3}_{key4}"

        return new_key

    def update_prefix(self, model_name, prefix, prefix_to_save, prefix_all, pytorch_state_dict):
        """
        改模型 参数前缀

        :param model_name:
        :param prefix: 前缀
        :param prefix_to_save: 替换的前缀
        :param prefix_all: 所有的前缀
        :param pytorch_state_dict:
        :return:
        """
        if prefix_all is not None and len(prefix_all) == 0:
            prefix_all = self.get_model_prefix_pattern(model_name=model_name, name="prefix_all")

        if prefix is not None and len(prefix) == 0:
            prefix = self.get_model_prefix_pattern(model_name=model_name, name="prefix")

        if prefix_to_save is not None and len(prefix_to_save) == 0:
            prefix_to_save = self.get_model_prefix_pattern(model_name=model_name, name="prefix_to_save")

        # 改前缀
        save_pytorch_state_dict = OrderedDict()
        for k, v in pytorch_state_dict.items():
            new_key = k
            if isinstance(prefix, list):
                for index, item in enumerate(prefix):
                    if isinstance(prefix_to_save, list) and len(prefix_to_save) > index:
                        new_item = prefix_to_save[index]
                    else:
                        new_item = ""
                    if str(k).startswith(item):
                        new_key = str(k).replace(item, new_item)
                    else:
                        new_key = k
            else:
                if str(k).startswith(prefix):
                    new_key = str(k).replace(prefix, prefix_to_save)

            prefix_new_key = f"{prefix_all}{new_key}"
            save_pytorch_state_dict[prefix_new_key] = v
        return save_pytorch_state_dict

    def yolov3_replace(self, model_name, key=None):
        """
        检测是否需要转换  layers{}_stage{}_conv_layer

            backbone.stage.1 -> backbone.stage_1
            backbone.stage.1.stage.1.1. -> backbone.stage_1.stage_1_1
            neck.yolo_block.0 -> neck.yolo_block_0
            neck.yolo_transition.0  -> neck.yolo_transition_0
            yolo_head.yolo_output.2  -> yolo_head.yolo_output_2.
        :param model_name:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            key4 = match_res[3]
            new_key = f"{key1}_{key2}_{key3}.{key4}"

        res_1 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[1])
        if len(res_1) > 0:
            match_res = res_1[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            key4 = match_res[3]
            new_key = f"{key1}_{key2}_{key3}.{key4}"

        res_2 = MatchUtils.match_pattern_extract(new_key, pattern=pattern_list[2])
        if len(res_2) > 0:
            match_res = res_2[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}.{key3}"

        return new_key

    def yolov5_replace(self, model_name, key=None):
        """
        检测是否需要转换  yolo_output_{}

            yolo_head.yolo_output.0.weight -> yolo_head.yolo_output_0.weight

        :param model_name:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}.{key3}"

        return new_key

    def yolov6_replace(self, model_name, key=None):
        """
        检测是否需要转换 stage{}_simsppf

            backbone.stage2.repconv.rbr_dense.0.weight -> backbone.stage2_repconv.rbr_dense.0.weight

        :param model_name:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}{key3}"

        return new_key

    def yolov7_replace(self, model_name, key=None):
        """
        检测是否需要转换 stage{}_simsppf
            re.compile(r"(backbone.layers\d+)\.(stage\d+)\.(.*)")
            backbone.layers4.stage1.elan_layer.conv1.conv.weight ->backbone.layers4_stage1_elan_layer.conv1.conv.weight

        :param model_name:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}_{key3}"

        res_1 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[1])
        if len(res_1) > 0:
            match_res = res_1[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}{key3}"

        return new_key

    def rtmdet_replace(self, model_name, key=None):
        """
        检测是否需要转换

            backbone.layers2.stage1.cspnext_layer.conv1.conv.weight -> backbone.layers2_stage1_cspnext_layer.conv1.conv.weight

        :param model_name:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}_{key3}"

        return new_key

    def yolov8_replace(self, model_name, key=None):
        """
        检测是否需要转换 stage{}_simsppf
            re.compile(r"(backbone.layers\d+)\.(stage\d+)\.(.*)")
            backbone.layers4.stage1.elan_layer.conv1.conv.weight ->backbone.layers4_stage1_elan_layer.conv1.conv.weight
            backbone.layers2.stage1.c2f_layer.conv1.conv.weight ->  backbone.layers2_stage1_c2f_layer.conv1.conv.weight
        :param model_name:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            new_key = f"{key1}_{key2}_{key3}"

        return new_key

    def ocr_db_replace(self, model_name, key=None):
        """
        检测是否需要转换  layers{}_stage{}_conv_layer

            backbone.stage0.0.expand_conv.conv.weight -> backbone.stages.0.0.expand_conv.conv.weight
        :param model_name:
        :param key:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            key3 = match_res[2]
            key4 = match_res[3]
            new_key = f"{key1}s.{key2}.{key3}.{key4}"

        return new_key

    def ocr_crnn_replace(self, model_name, key=None):
        """
        检测是否需要转换  layers{}_stage{}_conv_layer

            Student.backbone.conv1._conv.weight -> backbone.conv1._conv.weight
        :param model_name:
        :param key:
        :return:
        """
        pattern_list = self.get_model_class_pattern(model_name)
        if key is None or len(key) == 0:
            return key

        new_key = key

        res_0 = MatchUtils.match_pattern_extract(key, pattern=pattern_list[0])
        if len(res_0) > 0:
            match_res = res_0[0]
            key1 = match_res[0]
            key2 = match_res[1]
            new_key = f"{key2}"

        return new_key

    def check_model_name_replace(self, model_name, key=None):
        """
        检查模型参数需要变化

        :param model_name:
        :param key:
        :return:
        """

        if self.check_need_number_replace(model_name=model_name, key=key):
            key = self.number_replace(key)

        # yolox
        if self.check_need_csp_darknet_replace(model_name=model_name, key=key):
            key = self.csp_darknet_replace(key)

        # yolov3 , DB, (OCR) CRNN (OCR) , yolov5, yolov6, yolov7, yolov8
        if MatchUtils.match_pattern_list_flag(texts=key, pattern_list=self.get_model_class_pattern(model_name)):
            key = self.do_model_layer_name_replace(model_name=model_name, key=key)

        return key

    def do_model_layer_name_replace(self, model_name=None, key=None):
        """
        进行 模型参数变化

        :param model_name:
        :param key:
        :return:
        """
        model_class = self.get_model_class(model_name)

        if model_class in ["yolov3"]:
            key = self.yolov3_replace(model_name=model_name, key=key)
        elif model_class in ["DB"]:
            key = self.ocr_db_replace(model_name=model_name, key=key)
        elif model_class in ["CRNN"]:
            key = self.ocr_crnn_replace(model_name=model_name, key=key)
        elif model_class in ["yolov5", "yolov5p6"]:
            key = self.yolov5_replace(model_name=model_name, key=key)
        elif model_class in ["yolov6"]:
            key = self.yolov6_replace(model_name=model_name, key=key)
        elif model_class in ["yolov7", "yolov7p6"]:
            key = self.yolov7_replace(model_name=model_name, key=key)
        elif model_class in ["rtmdet"]:
            key = self.rtmdet_replace(model_name=model_name, key=key)
        elif model_class in ["yolov8", "yolov8p6"]:
            key = self.yolov8_replace(model_name=model_name, key=key)

        return key

    def check_model_parameter_transpose(self, model_name, key=None, weight_value=None, show_info=False):
        """
        检查模型参数需要 转置

        :param model_name:
        :param key:
        :param weight_value:
        :param show_info:
        :return:
        """
        # if k[-7:] == ".weight":
        #     if ".embeddings." not in k \
        #             and ".layer_norm." not in k \
        #             and "._se.conv" not in k \
        #             and "conv.weight" not in k \
        #             and "head.head_cls" not in k \
        #             and "._conv." not in k:
        #         new_weight_value = new_weight_value.transpose(0, -1)
        #         if show_info:
        #             logger.info('=' * 20, '[transpose]', k, '=' * 20)

        if MatchUtils.match_pattern_list_flag(texts=key, pattern_list=self.get_model_weight_pattern(model_name)):
            weight_value = weight_value.transpose(0, -1)
            if show_info:
                logger.info('=' * 20, '[transpose]', key, '=' * 20)

        return weight_value

    def check_model_parameter_filter(self, model_name, key=None, weight_value=None, show_info=False):
        """
        检查模型参数需要 过滤

        :param model_name:
        :param key:
        :param weight_value:
        :param show_info:
        :return:
        """
        flag = MatchUtils.match_pattern_list_flag(texts=key, pattern_list=self.get_model_filter_pattern(model_name))

        return flag

    def transform_to_torch(self, in_model_dir, output_dir,
                           prefix="",
                           prefix_to_save="",
                           prefix_all="",
                           paddle_to_torch_param_name=None,
                           filter_param_name=None,
                           show_info=True,
                           do_transform=False,
                           model_name=None):
        """
        转换DPR CK

        :param in_model_dir:
        :param output_dir:
        :param prefix:
        :param prefix_to_save:
        :param prefix_all:
        :param paddle_to_torch_param_name:
        :param filter_param_name:
        :param show_info:
        :param do_transform:
        :param model_name:
        :return:
        """
        if not do_transform and FileUtils.check_file_exists(output_dir):
            logger.info(f"模型已存在：{output_dir}")
            return output_dir, []

        if in_model_dir is not None and str(in_model_dir).endswith(".pdparams"):
            in_model_file_name = in_model_dir
        else:
            in_model_file_name = f"{in_model_dir}/model_state.pdparams"
        in_model_vocab = f"{in_model_dir}/vocab.json"
        paddle_model_params, paddle_model_show_params = self.load_paddle_model(in_model_file_name, show_info=False)

        paddle_to_torch = {
            "._mean": ".running_mean",
            "._variance": ".running_var",
            ".embeddings.layer_norm.": ".embeddings.LayerNorm.",
        }

        pytorch_state_dict = OrderedDict()

        if paddle_to_torch_param_name is not None:
            paddle_to_torch.update(paddle_to_torch_param_name)

        model_class = self.get_model_class(model_name)
        logger.info(f"开始参数转换：model_name: {model_name} - model_class: {model_class}")

        # 开始转换
        for k, v in paddle_model_params.items():

            # 过滤不必要的参数
            if (filter_param_name is not None and k in filter_param_name) \
                    or self.check_model_parameter_filter(model_name=model_name, key=k):
                if show_info:
                    logger.info(f'过滤不必要的参数：{k} - {v.shape}')
                continue

            old_k = k
            new_weight_value = torch.FloatTensor(v)

            for paddle_name, torch_name in paddle_to_torch.items():
                k = k.replace(paddle_name, torch_name)

                k = self.check_model_name_replace(model_name=model_name, key=k)

            # 模型参数维度转置
            new_weight_value = self.check_model_parameter_transpose(model_name=model_name,
                                                                    key=k,
                                                                    weight_value=new_weight_value)

            pytorch_state_dict[k] = new_weight_value
            if show_info:
                logger.info(f"{old_k} -> {k} - {v.shape}")

        save_pytorch_state_dict = self.update_prefix(model_name=model_name,
                                                     prefix=prefix,
                                                     prefix_to_save=prefix_to_save,
                                                     prefix_all=prefix_all,
                                                     pytorch_state_dict=pytorch_state_dict)

        # 添加其他
        model_class = self.get_model_class(model_name)
        model_add_pattern = self.get_model_add_pattern(model_name=model_class)
        if len(model_add_pattern) > 0:
            if model_class in ["yolov8", "yolov8p6"]:
                save_pytorch_state_dict[model_add_pattern[0]] = torch.tensor([1.0])

        if output_dir is not None and str(output_dir).rfind(".") > -1:
            save_model_path = output_dir
        else:
            save_model_path = os.path.join(output_dir, "pytorch_model.bin")
        # if show_info:
        logger.info(f"转换paddle模型为pytorch: {in_model_file_name} - {save_model_path} - "
                    f"{len(paddle_model_params.keys())} -> {len(save_pytorch_state_dict)}")

        FileUtils.check_file_exists(save_model_path)
        torch.save(save_pytorch_state_dict, save_model_path)
        # FileUtils.copy_file(in_model_vocab, output_dir)

        return save_model_path, paddle_model_show_params

    def get_filter_param_name(self, model_name):
        """
        获取 filter parameters

        :param model_name:
        :return:
        """
        filter_param_name = [
            "head.p7_feat.scale_reg"
        ]

        # if model_name in ["ppyoloe_crn_s_300e_coco"]:
        if str(model_name).startswith("ppyoloe"):
            filter_param_name = [
                "yolo_head.anchor_points",
                "yolo_head.stride_tensor"
            ]
        if str(model_name).lower == 'CRNN':
            filter_param_name = [
                "yolo_head.anchor_points",
                "yolo_head.stride_tensor"
            ]

        return filter_param_name

    def transform(self, in_model_dir, output_dir=None,
                  prefix="",
                  prefix_to_save="",
                  prefix_all="",
                  paddle_to_torch_param_name=None,
                  filter_param_name=None,
                  show_info=True,
                  do_transform=False,
                  model_name=None):

        """
        转换模型参数

        :param in_model_dir:
        :param output_dir:
        :param prefix:
        :param prefix_to_save:
        :param prefix_all:
        :param paddle_to_torch_param_name:
        :param filter_param_name:
        :param show_info:
        :param do_transform:
        :param model_name:
        :return:
        """
        if model_name is None:
            model_name = FileUtils.get_file_name(in_model_dir)

        if output_dir is None:
            model_path = FileUtils.get_dir_file_name(in_model_dir)
            # paddlenlp model
            find_flag = self.find_some_name_in(model_name)

            if str(in_model_dir).find(".paddlenlp") > -1 or (model_name is not None and find_flag):
                output_model_name = f"pytorch_model.bin"
                FileUtils.copy_file_rename(f"{model_path}/model_config.json", f"{model_path}/config.json")
            else:
                output_model_name = f"{model_name}.pth"
            output_dir = f"{model_path}/{output_model_name}"

        if filter_param_name is None:
            filter_param_name = self.get_filter_param_name(model_name=model_name)

        save_model_path, paddle_model_show_params = self.transform_to_torch(in_model_dir=in_model_dir,
                                                                            output_dir=output_dir,
                                                                            prefix=prefix,
                                                                            prefix_to_save=prefix_to_save,
                                                                            prefix_all=prefix_all,
                                                                            paddle_to_torch_param_name=paddle_to_torch_param_name,
                                                                            filter_param_name=filter_param_name,
                                                                            show_info=show_info,
                                                                            do_transform=do_transform,
                                                                            model_name=model_name)

        return save_model_path, paddle_model_show_params

    def find_some_name_in(self, model_name):
        """
        查找部分存在

        :param model_name:
        :return:
        """
        need_rename_list = ["LayoutXLM", "Ernie"]
        find_flag = False
        for name in need_rename_list:
            if str(model_name).find(name) > -1:
                find_flag = True
                break
        return find_flag


def demo_convert():
    # model_name = "picodet_s_320_coco"
    model_name = "picodet_s_416_coco"

    paddle_weight_base_dir = f"~/.cache/paddle/weights"
    file_name = f"{paddle_weight_base_dir}/{model_name}.pdparams"
    output_dir = f"{paddle_weight_base_dir}/{model_name}.pth"
    base_dir = file_name

    convert = ConvertPaddleDetectionModelToTorch()
    convert.transform(in_model_dir=base_dir, output_dir=None, prefix="",
                      prefix_to_save="",
                      prefix_all="",
                      paddle_to_torch_param_name=None,
                      filter_param_name=None)


if __name__ == '__main__':
    demo_convert()
