# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WTDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    #14
    METAINFO = dict(
        classes=('background', 'mound', 'light', 'build', 'wall',
                 'steel', 'tank','cement', 'boat', 'tower',
                 'jvma','Stone', 'tree', 'river'
                 ),
        palette=[[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,30], [220, 220, 0],[107, 142, 35],
                 [152, 251, 152], [70, 130, 180],[220, 20, 60], [255, 0, 0]
                 ])

    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',   # 标注mask图像的格式
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
