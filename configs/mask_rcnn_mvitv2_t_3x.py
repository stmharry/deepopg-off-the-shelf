import warnings

from fvcore.common.param_scheduler import MultiStepParamScheduler
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import constants  # noqa
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import lr_multiplier  # noqa
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import model  # noqa
from projects.MViTv2.configs.mask_rcnn_mvitv2_t_3x import dataloader, optimizer, train

from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.solver import WarmupParamScheduler

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


dataloader.train.dataset = L(get_detection_dataset_dicts)(names="pano_train")
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = 1

dataloader.test.dataset = L(get_detection_dataset_dicts)(names="pano_eval")
dataloader.test.mapper.instance_mask_format = "${...train.mapper.instance_mask_format}"


# train
train.max_iter = 67500

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[52500, 62500, 67500],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# optimizer

optimizer.lr = 1.6e-4

# from omegaconf import OmegaConf; d = OmegaConf.to_container(cfg)
# pp d
