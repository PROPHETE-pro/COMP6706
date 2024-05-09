import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



@DATASETS.register_module()
class COCODataset_4(CustomDataset):
    """COCO dataset.
    """

    
    def __init__(self, **kwargs):
        super(COCODataset_4, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_614.png',
            reduce_zero_label=False, 
            **kwargs)
