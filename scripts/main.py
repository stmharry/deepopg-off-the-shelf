import importlib

from absl import logging

from app.data import InstanceDetectionV1
from detectron2.engine import default_argument_parser, launch


def main():
    parser = default_argument_parser()
    parser.add_argument("--main-app")
    parser.add_argument("--data-dir")
    parser.add_argument("--EVAL-FLAG", type=int, default=1)
    args = parser.parse_args()

    logging.info(f"Command Line Args: {args!s}")

    InstanceDetectionV1.register(root_dir=args.data_dir)

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
