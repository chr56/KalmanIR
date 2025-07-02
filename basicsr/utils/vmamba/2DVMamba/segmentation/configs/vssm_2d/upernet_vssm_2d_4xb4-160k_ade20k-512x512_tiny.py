_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
# checkpoint_path = ('/scratch/KurcGroup/jingwei/result/v2dmamba/vmamba/'
#                    'vmambav2v_2d_tiny_224/vssm1_tiny_0230s/20241023030251/'
#                    'ckpt_epoch_269.pth')  # noqa
# checkpoint_path = ('/gpfs/scratch/jingwezhang/checkpoint/v2dmamba/tiny_1k/ckpt_epoch_269.pth')  # noqa
checkpoint_path = ('')
model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=checkpoint_path, # here is the path
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        # depths=(2, 2, 5, 2),
        depths=(2, 2, 8, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        # ssm_ratio=2.0,
        ssm_ratio=1.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
        norm_layer="ln2d",
        use_v2d=True
    ),)
train_dataloader = dict(batch_size=4) # as gpus=4

# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='SegVisualizationHook', draw=True, interval=1))


