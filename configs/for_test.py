manipulate_arch = False
model = dict(
    type='DynamicEncoderDecoder',
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=32,
        body_depth=[2, 2, 4, 2],
        body_width=[48, 96, 192, 384],
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        conv_cfg=dict(type='DynConv2d'),
        norm_cfg=dict(type='DynSyncBN', requires_grad=True),
        style='pytorch'),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=1536,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=192,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=dict(type='DynSyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='DynamicFCNHead',
        conv_cfg=dict(type='DynConv2d'),
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
train_cfg = dict()
test_cfg = dict(mode='whole')
albu_train_transforms = [dict(type='RandomRotate90', p=0.3)]
cityscapes_dataset_train = dict(
    unbalanced=True,
    category_num=3,
    type='CityscapesDataset_4',
    data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/cityscapes/',
    img_dir='leftImg8bit/train',
    ann_dir='gtFine/train',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=4, extent=20, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
kitti_dataset_train = dict(
    unbalanced=True,
    category_num=3,
    type='KittiDataset_4',
    data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/KITTI',
    img_dir='training/image_2',
    ann_dir='training/semantic',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=4, extent=20, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
ADEChanllenge_dataset_train = dict(
    unbalanced=True,
    category_num=3,
    type='ADE20KDataset_normal',
    data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data',
    img_dir='cleaned_ade/images/training',
    ann_dir='ade/ADEChallengeData2016/annotations/training',
    label_map=dict({
        13: 1,
        3: 2,
        18: 3,
        10: 3,
        5: 3,
        4: 255,
        17: 255,
        14: 255,
        30: 255,
        35: 255
    }),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            label_map=dict({
                13: 1,
                3: 2,
                18: 3,
                10: 3,
                5: 3,
                4: 255,
                17: 255,
                14: 255,
                30: 255,
                35: 255
            })),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=2, extent=5, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
Viper_dataset_train = dict(
    type='ViperDataset_4',
    category_num=3,
    data_root=
    '/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/cleaned_VIPER/VIPER',
    img_dir='train_img_00-77_0_jpg/train/img',
    ann_dir='train_cls_00-77_0/train/cls',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=4, extent=20, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
wild_dash_train = dict(
    type='WildDashDataset_4',
    category_num=3,
    unbalanced=True,
    data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/wilddash',
    img_dir='images',
    ann_dir='labels/labels',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=4, extent=20, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
coco_dataset_train = dict(
    type='COCODataset_4',
    category_num=3,
    unbalanced=True,
    data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/COCO',
    img_dir='train2017',
    ann_dir='semantic_train2017',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
cloth_dataset = dict(
    type='Pseudo_clothes_person_Dataset',
    category_num=3,
    unbalanced=False,
    data_root='/mnt/diske/qing_chang/Data/clothes_person_pseudo/',
    img_dir='img',
    ann_dir='label',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=2, extent=5, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
wall_dataset = dict(
    type='wall_dataset',
    category_num=3,
    unbalanced=False,
    data_root=
    '/mnt/diskg/ruitong_gan/gaiaseg_huawei_pack/GAIA-seg-dev/wall_final/',
    img_dir='img',
    ann_dir='annotation',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
green_dataset = dict(
    type='green_dataset',
    category_num=3,
    unbalanced=False,
    data_root='/mnt/diskg/ruitong_gan/gaiaseg_huawei_pack/GAIA-seg-dev/green/',
    img_dir='img',
    ann_dir='annotation',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
    
repeated_green = dict(
    type='RepeatDataset',
    times=5,
    dataset=green_dataset)

mix_dataset = dict(
    type='mix_dataset',
    category_num=3,
    unbalanced=False,
    data_root=
    '/mnt/diskg/ruitong_gan/gaiaseg_huawei_pack/GAIA-seg-dev/mix_final/',
    img_dir='img',
    ann_dir='annotation',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
    
repeated_mix = dict(
    type='RepeatDataset',
    times=15,
    dataset=mix_dataset)
        
greenwall_dataset = dict(
    type='greenwall_dataset',
    category_num=3,
    unbalanced=False,
    data_root=
    '/mnt/diskg/ruitong_gan/gaiaseg_huawei_pack/GAIA-seg-dev/greenwall_final/',
    img_dir='img',
    ann_dir='annotation',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
indoor_dataset = dict(
    type='indoor_dataset',
    category_num=3,
    unbalanced=False,
    data_root=
    '/mnt/diskg/ruitong_gan/gaiaseg_huawei_pack/GAIA-seg-dev/indoor_final/',
    img_dir='img',
    ann_dir='annotation',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
repeated_indoor = dict(
    type='RepeatDataset',
    times=15,
    dataset=indoor_dataset)
    
repeated_greenwall = dict(
    type='RepeatDataset',
    times=5,
    dataset=greenwall_dataset)
    
coco_dataset_val = dict(
    type='COCODataset_4',
    category_num=3,
    unbalanced=True,
    data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/COCO/',
    img_dir='val2017',
    ann_dir='semantic_val2017',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
        dict(type='Albu', transforms=[dict(type='RandomRotate90', p=0.3)]),
        dict(type='MotionBlur', min_extent=3, extent=10, ez_blur=True),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
mapillary_dataset = dict(
    type='MapillaryDataset_4',
    category_num=3,
    data_root='/mnt/diske/qing_chang/Data/Mapillary-Street/training',
    img_dir='images',
    ann_dir='v2.0/labels',
    label_map=dict({
        30: 1,
        31: 1,
        32: 1,
        33: 1,
        34: 1,
        61: 2,
        64: 3,
        63: 255
    }),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            label_map=dict({
                30: 1,
                31: 1,
                32: 1,
                33: 1,
                34: 1,
                61: 2,
                64: 3,
                63: 255
            })),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
bdd_dataset = dict(
    type='BDDDataset_4',
    category_num=3,
    unbalanced=True,
    data_root='/mnt/diske/qing_chang/Data/BDD100K/bdd100k/seg',
    img_dir='images/train',
    ann_dir='labels/train',
    label_map=dict({
        8: 3,
        9: 3,
        10: 2,
        11: 1,
        12: 1,
        0: 255
    }),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            label_map=dict({
                8: 3,
                9: 3,
                10: 2,
                11: 1,
                12: 1,
                0: 255
            })),
        dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(512, 1024), pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg'])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        coco_dataset_val,
        #coco_dataset_train,
        cityscapes_dataset_train,
        kitti_dataset_train,
        #wild_dash_train,
        #Viper_dataset_train,
        ADEChanllenge_dataset_train,
        mapillary_dataset,
        #bdd_dataset,
        #cloth_dataset,
        #wall_dataset,
        repeated_greenwall,
        #green_dataset,
        #repeated_green,
        #greenwall_dataset
        #repeated_indoor,
        repeated_mix,
    ],
    val=dict(
        type='CityscapesDataset_4',
        data_root=
        '/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/cityscapes',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset_4',
        data_root='/mnt/diskb/qing_chang/env_mmlab/mmsegmentation/data/cityscapes',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './change_head_overfit_green/iter_30000.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.0015, momentum=0.9, weight_decay=0.0005,
                  paramwise_cfg = dict(
                                     custom_keys={'head':dict(lr_mult=10.)}))
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=2000, metric='mIoU')
work_dir = './change_head_overfit_green'
gpu_ids = range(0, 1)
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
                dict({
                    'name': 'MAX',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [80, 160, 320, 640],
                    'arch.backbone.body.depth': [4, 6, 29, 4]
                }),
                dict({
                    'name': 'MIN',
                    'arch.backbone.stem.width': 32,
                    'arch.backbone.body.width': [48, 96, 192, 384],
                    'arch.backbone.body.depth': [2, 2, 4, 2]
                })
            ]),
        dict(
            type='flops',
            key=
            '/mnt/diske/qing_chang/GAIA/GAIA-seg-dev/hubs/flops/56_flops.json',
            input_size='3,512,1024',
            bin_num=5)
    ])
val_sampler = dict(
    type='anchor',
    anchors=[
        dict({
            'name': 'R50',
            'arch.backbone.stem.width': 32,
            'arch.backbone.body.width': [48, 96, 192, 384],
            'arch.backbone.body.depth': [2, 2, 4, 2]
        })
    ])
