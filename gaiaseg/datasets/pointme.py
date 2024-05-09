from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PointMeDataset(CustomDataset):
    """
    Pointme dataset
    """

    CLASSES = ('background','person','sky','vegetation')

    PALETTE = [[0,0,0],[220, 20, 60],[70, 130, 180],[107, 142, 35]]
    
    def __init__(self, **kwargs):
        super(PointMeDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_614.png',
            reduce_zero_label=False, 
            **kwargs)
