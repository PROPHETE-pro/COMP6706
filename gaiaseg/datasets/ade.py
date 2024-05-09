from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



@DATASETS.register_module()
class ADE20KDataset_4(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """


    
    def __init__(self, **kwargs):
        super(ADE20KDataset_4, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_614.png',
            #reduce_zero_label=False, # 还好这个bug是在用ade val验证的时候才会出现
            **kwargs)
