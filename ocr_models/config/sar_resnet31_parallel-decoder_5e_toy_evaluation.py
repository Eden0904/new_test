toy_data_root = 'data/text_reog'
toy_rec_train = dict(
    type='OCRDataset',
    data_root='data/text_reog',
    ann_file='train_label(1).json',
    pipeline=None,
    test_mode=False)
toy_rec_test = dict(
    type='OCRDataset',
    data_root='data/text_reog',
    ann_file='test_label(1).json',
    pipeline=None,
    test_mode=True)
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
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
load_from = '/root/workspace/ocr_models/config/epoch_10.pth'
resume = False
val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ],
    dataset_prefixes=['ours'])
test_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ],
    dataset_prefixes=['ours'])
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextRecogLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', lr=0.001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [dict(type='MultiStepLR', milestones=[3, 4], end=5)]
dictionary = dict(
    type='Dictionary',
    dict_file=
    '/root/workspace/ocr_models/config/chinese_english_digits.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True)
model = dict(
    type='SARNet',
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[127, 127, 127],
        std=[127, 127, 127]),
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='SAREncoder', enc_bi_rnn=False, enc_do_rnn=0.1, enc_gru=False),
    decoder=dict(
        type='ParallelSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True,
        postprocessor=dict(type='AttentionPostprocessor'),
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, reduction='mean'),
        dictionary=dict(
            type='Dictionary',
            dict_file=
            '/root/workspace/ocr_models/config/chinese_english_digits.txt',
            with_start=True,
            with_end=True,
            same_start_end=True,
            with_padding=True,
            with_unknown=True),
        max_seq_len=30))
train_pipeline = [
    dict(type='LoadImageFromFile', ignore_empty=True, min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RescaleToHeight',
        height=48,
        min_width=48,
        max_width=160,
        width_divisor=4),
    dict(type='PadToWidth', width=160),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type':
            'ConditionApply',
            'true_transforms': [{
                'type':
                'ImgAugWrapper',
                'args': [{
                    'cls': 'Rot90',
                    'k': 0,
                    'keep_size': False
                }]
            }],
            'condition':
            "results['img_shape'][1]<results['img_shape'][0]"
        }, {
            'type':
            'ConditionApply',
            'true_transforms': [{
                'type':
                'ImgAugWrapper',
                'args': [{
                    'cls': 'Rot90',
                    'k': 1,
                    'keep_size': False
                }]
            }],
            'condition':
            "results['img_shape'][1]<results['img_shape'][0]"
        }, {
            'type':
            'ConditionApply',
            'true_transforms': [{
                'type':
                'ImgAugWrapper',
                'args': [{
                    'cls': 'Rot90',
                    'k': 3,
                    'keep_size': False
                }]
            }],
            'condition':
            "results['img_shape'][1]<results['img_shape'][0]"
        }],
                    [{
                        'type': 'RescaleToHeight',
                        'height': 48,
                        'min_width': 48,
                        'max_width': 160,
                        'width_divisor': 4
                    }], [{
                        'type': 'PadToWidth',
                        'width': 160
                    }], [{
                        'type': 'LoadOCRAnnotations',
                        'with_text': True
                    }],
                    [{
                        'type':
                        'PackTextRecogInputs',
                        'meta_keys':
                        ('img_path', 'ori_shape', 'img_shape', 'valid_ratio')
                    }]])
]
train_list = [
    dict(
        type='OCRDataset',
        data_root='data/text_reog',
        ann_file='train_label(1).json',
        pipeline=None,
        test_mode=False)
]
test_list = [
    dict(
        type='OCRDataset',
        data_root='data/text_reog',
        ann_file='test_label(1).json',
        pipeline=None,
        test_mode=True)
]
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                data_root='data/text_reog',
                ann_file='train_label(1).json',
                pipeline=None,
                test_mode=False)
        ],
        pipeline=[
            dict(type='LoadImageFromFile', ignore_empty=True, min_size=2),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                type='RescaleToHeight',
                height=48,
                min_width=48,
                max_width=160,
                width_divisor=4),
            dict(type='PadToWidth', width=160),
            dict(
                type='PackTextRecogInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'valid_ratio'))
        ]))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                data_root='data/text_reog',
                ann_file='test_label(1).json',
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RescaleToHeight',
                height=48,
                min_width=48,
                max_width=160,
                width_divisor=4),
            dict(type='PadToWidth', width=160),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                type='PackTextRecogInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'valid_ratio'))
        ]))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                data_root='data/text_reog',
                ann_file='test_label(1).json',
                pipeline=None,
                test_mode=True)
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RescaleToHeight',
                height=48,
                min_width=48,
                max_width=160,
                width_divisor=4),
            dict(type='PadToWidth', width=160),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                type='PackTextRecogInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'valid_ratio'))
        ]))
launcher = 'none'
work_dir = 'test_dir/batch32_Wordmetric/'