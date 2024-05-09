import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



@DATASETS.register_module()
class KittiDataset_4(CustomDataset):


    CLASSES = ('background','person','sky','vegetation')
    PALETTE = [[0,0,0],[220, 20, 60],[70, 130, 180],[107, 142, 35]]

    def __init__(self, **kwargs):
        super(KittiDataset_4, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_614.png', 
            **kwargs) 


