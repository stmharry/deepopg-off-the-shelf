import warnings

from fvcore.common.param_scheduler import MultiStepParamScheduler
from projects.MViTv2.configs.common.coco_loader import dataloader
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import constants  # noqa
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import model  # noqa
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import optimizer, train

from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.solver import WarmupParamScheduler

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# dataloader

dataloader.train.dataset = L(get_detection_dataset_dicts)(names="pano_train")
dataloader.train.mapper.augmentations = [
    L(T.RandomContrast)(intensity_min=0.7, intensity_max=1.3),
    L(T.RandomBrightness)(intensity_min=0.7, intensity_max=1.3),
    L(T.ResizeShortestEdge)(short_edge_length=(768, 1280), sample_style="range"),
    L(T.RandomCrop)(crop_size=(384, 640), crop_type="absolute_range"),
]
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = 4
dataloader.train.num_workers = 12

dataloader.test.dataset = L(get_detection_dataset_dicts)(names="pano_eval")
dataloader.test.mapper.instance_mask_format = "${...train.mapper.instance_mask_format}"

# train

train.max_iter = 100_000
train.eval_period = 10_000
train.log_period = 20

train.checkpointer.max_to_keep = 100
train.checkpointer.period = 10_000

# optimizer

optimizer.lr = 1e-3

# lr_multiplier

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[70_000, 80_000, 90_000],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)
