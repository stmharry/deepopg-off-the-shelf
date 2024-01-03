import importlib
import warnings

from absl import logging

from app.instance_detection.datasets import InstanceDetection
from app.semantic_segmentation.datasets import SemanticSegmentation
from detectron2.engine import default_argument_parser, launch

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def main():
    logging.set_verbosity(logging.INFO)

    parser = default_argument_parser()
    parser.add_argument("--main-app")
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset_name")
    parser.add_argument("--EVAL-FLAG", type=int, default=1)
    args = parser.parse_args()

    logging.info(f"Command Line Args: {args!s}")

    for dataset_name in args.dataset_name.split(","):
        InstanceDetection.register_by_name(dataset_name, root_dir=args.data_dir)
        SemanticSegmentation.register_by_name(dataset_name, root_dir=args.data_dir)

    (module, name) = args.main_app.split(":")
    main_app = getattr(importlib.import_module(module), name)

    launch(
        main_app,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
