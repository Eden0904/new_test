model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_eval=False,
        style='caffe'),
    neck=dict(
        type='FPNC', in_channels=[64, 128, 256, 512], lateral_channels=256),
    det_head=dict(
        type='DBHead',
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=0.12549019607843137,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5], {
            'cls': 'Affine',
            'rotate': [-10, 10]
        }, ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
icdar2015_textdet_data_root = 'data/text_localization'
icdar2015_textdet_train = dict(
    type='OCRDataset',
    data_root='data/text_localization',
    ann_file='train_label.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True),
        dict(
            type='TorchVisionWrapper',
            op='ColorJitter',
            brightness=0.12549019607843137,
            saturation=0.5),
        dict(
            type='ImgAugWrapper',
            args=[['Fliplr', 0.5], {
                'cls': 'Affine',
                'rotate': [-10, 10]
            }, ['Resize', [0.5, 3.0]]]),
        dict(type='RandomCrop', min_side_ratio=0.1),
        dict(type='Resize', scale=(640, 640), keep_ratio=True),
        dict(type='Pad', size=(640, 640)),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape'))
    ])
icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root='data/text_localization',
    ann_file='test_label.json',
    test_mode=True,
    pipeline=[
        dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
        dict(type='Resize', scale=(1333, 736), keep_ratio=True),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ])
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = None
resume = False
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='ConstantLR', factor=1.0)]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRDataset',
        data_root='data/text_localization',
        ann_file='train_label.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5),
            dict(
                type='ImgAugWrapper',
                args=[['Fliplr', 0.5], {
                    'cls': 'Affine',
                    'rotate': [-10, 10]
                }, ['Resize', [0.5, 3.0]]]),
            dict(type='RandomCrop', min_side_ratio=0.1),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size=(640, 640)),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape'))
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/text_localization',
        ann_file='test_label.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(type='Resize', scale=(1333, 736), keep_ratio=True),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/text_localization',
        ann_file='test_label.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type='color_ignore_orientation'),
            dict(type='Resize', scale=(1333, 736), keep_ratio=True),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
auto_scale_lr = dict(base_batch_size=4)
launcher = 'none'
work_dir = './work_dirs\\dbnet_resnet18_fpnc_1200e_icdar2015'
