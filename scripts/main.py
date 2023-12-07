import importlib
import warnings

from absl import logging

from app.semantic_segmentation.datasets import SemanticSegmentationV4
from detectron2.engine import default_argument_parser, launch

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def main():
    logging.set_verbosity(logging.INFO)

    parser = default_argument_parser()
    parser.add_argument("--main-app")
    parser.add_argument("--data-dir")
    parser.add_argument("--EVAL-FLAG", type=int, default=1)
    args = parser.parse_args()

    logging.info(f"Command Line Args: {args!s}")

    # InstanceDetectionV1.register(root_dir=args.data_dir)
    # InstanceDetectionV1NTUH.register(root_dir=args.data_dir)
    SemanticSegmentationV4.register(root_dir=args.data_dir)

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
