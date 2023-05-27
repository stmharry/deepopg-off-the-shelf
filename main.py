import logging

from detectron2.engine import default_argument_parser, launch

# this is from MaskDINO
from train_net import main

from app.data import InstanceDetectionV1

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = InstanceDetectionV1(root_dir="/mnt/hdd/PANO/data/")
    dataset.prepare_coco(n_jobs=-1)
    dataset.load()

    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--EVAL_FLAG", type=int, default=1)
    args = parser.parse_args()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
