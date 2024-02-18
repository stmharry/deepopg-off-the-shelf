from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
    InstanceDetectionPredictionList,
)
from app.masks import Mask
from app.semantic_segmentation.datasets import SemanticSegmentation
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Detection Dataset name.")
flags.DEFINE_string(
    "input_prediction_name", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_string(
    "semseg_result_dir", None, "Semantic segmentation result directory."
)
flags.DEFINE_string("semseg_dataset_name", None, "Semantic segmentation dataset name.")
flags.DEFINE_string(
    "semseg_prediction_name",
    "inference/sem_seg_predictions.json",
    "Input prediction file name.",
)

flags.DEFINE_bool(
    "use_gt_as_prediction",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground truth "
    "finding summary but only ground truth segmentation.",
)
flags.DEFINE_string(
    "output_prediction_name",
    "instances_predictions.postprocessed.pth",
    "Output prediction file name.",
)
flags.DEFINE_string("csv_name", "result.csv", "Output result file name.")
flags.DEFINE_float("min_score", 0.01, "Confidence score threshold.")
flags.DEFINE_float("min_area", 0, "Object area threshold.")
flags.DEFINE_integer(
    "tooth_distance",
    150,
    "Implant has to be within this distance to be considered valid.",
)

FLAGS = flags.FLAGS


def basic_filter(
    df: pd.DataFrame,
    min_score: float,
    min_area: float,
) -> pd.DataFrame:
    _df = df.loc[(df["area"] > min_area) & (df["score"] > min_score)]

    logging.info(
        f"Filtering by area and score, reducing instances from {len(df)} -> {len(_df)}"
    )
    return _df


def calculate_mean_prob(
    mask: np.ndarray,
    prob: np.ndarray,
    bbox: np.ndarray | None = None,
    ignore_background: bool = False,
) -> np.ndarray:
    if bbox:
        x, y, w, h = bbox
        mask = mask[y : y + h + 1, x : x + w + 1]
        prob = prob[y : y + h + 1, x : x + w + 1]

    mean_prob = prob[mask].mean(axis=0)

    if ignore_background:
        mean_prob = np.r_["-1,1,0", 0, mean_prob[1:]]
        mean_prob = mean_prob / mean_prob.sum()

    return mean_prob


def main(_):
    logging.set_verbosity(logging.INFO)

    # instance detection

    data_driver: InstanceDetection | None = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    if data_driver is None:
        raise ValueError(f"Unknown dataset name: {FLAGS.dataset_name}")

    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    predictions: list[InstanceDetectionPrediction]
    if FLAGS.use_gt_as_prediction:
        predictions = [
            InstanceDetectionPrediction.from_instance_detection_data(data)
            for data in dataset
        ]

    else:
        predictions = InstanceDetectionPredictionList.from_detectron2_detection_pth(
            Path(FLAGS.result_dir, FLAGS.input_prediction_name)
        )

    id_to_prediction: dict[str | int, InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

    #

    semseg_data_driver: SemanticSegmentation | None = (
        SemanticSegmentation.register_by_name(
            dataset_name=FLAGS.semseg_dataset_name, root_dir=FLAGS.data_dir
        )
    )
    if semseg_data_driver is None:
        raise ValueError(f"Unknown dataset name: {FLAGS.semseg_dataset_name}")

    semseg_metadata: Metadata | None = MetadataCatalog.get(FLAGS.semseg_dataset_name)
    assert semseg_metadata is not None

    output_predictions: list[InstanceDetectionPrediction] = []
    row_results: list[dict[str, Any]] = []
    for data in dataset:
        # if data.file_name.stem != "cate4_0754932_20210719_PX_1_1":
        #     continue

        if data.image_id not in id_to_prediction:
            logging.warning(f"Image id {data.image_id} not found in predictions.")
            continue

        logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

        file_name: str = Path(data.file_name).relative_to(data_driver.image_dir).stem
        prediction: InstanceDetectionPrediction = id_to_prediction[data.image_id]
        instances: list[InstanceDetectionPredictionInstance] = prediction.instances

        df: pd.DataFrame = pd.DataFrame.from_records(
            [instance.dict() for instance in instances]
        )
        if len(df) == 0:
            continue

        logging.info(f"Original instance count: {len(df)}.")

        df["mask"] = df["segmentation"].map(
            lambda segmentation: Mask.from_obj(
                segmentation, width=data.width, height=data.height
            ).bitmask
        )
        df["area"] = df["mask"].map(np.sum)
        df["category_name"] = df["category_id"].map(metadata.thing_classes.__getitem__)
        df["is_tooth"] = df["category_name"].str.startswith("TOOTH")

        #

        df = basic_filter(df, min_score=FLAGS.min_score, min_area=FLAGS.min_area)
        if len(df) == 0:
            continue

        df_tooth = df.loc[df["is_tooth"]].copy()
        df_nontooth = df.loc[~df["is_tooth"]].copy()

        npz_path: Path = Path(
            FLAGS.semseg_result_dir, "inference", f"{data.file_name.stem}.npz"
        )
        # prob.shape == (num_classes, height, width)
        with np.load(npz_path) as npz:
            prob: np.ndarray = npz["prob"]

        prob = prob.transpose(1, 2, 0)
        prob = prob.astype(np.float32) / 65535

        # if there are overlaps, the probability of each pixel should be weakened becauses there is no non-maximum suppression
        mask_sum: np.array = np.sum(
            np.stack(df_tooth["mask"], axis=-1), axis=-1, keepdims=True
        )
        prob_modified: np.ndarray = 1 - np.power(1 - prob, 1 / mask_sum)

        df_tooth["score_per_tooth"] = None
        for index, row in df_tooth.iterrows():
            if not row["is_tooth"]:
                continue

            share_per_tooth = calculate_mean_prob(
                row["mask"],
                bbox=row["bbox"],
                prob=prob_modified,
                ignore_background=False,
            )
            df_tooth.at[index, "score_per_tooth"] = share_per_tooth

            # purely for debugging
            # semseg_category_id: int = np.argmax(share_per_tooth)
            # category_name = semseg_metadata.stuff_classes[semseg_category_id]
            # score: float = share_per_tooth[semseg_category_id]
            #
            # logging.debug(
            #     f"Reassigning instance of category {row['category_name']} with score {row['score']:.4f} "
            #     f"to category {category_name} with score {score:.4f}."
            # )
            #
            # df_tooth.at[index, "score_1"] = score
            # df_tooth.at[index, "category_name_1"] = category_name

        p_tooth: np.ndarray = np.stack(df_tooth["score_per_tooth"], axis=0)
        existence_score: np.ndarray = 1 - np.prod(1 - p_tooth, axis=0)

        s_missing: pd.Series = pd.Series(
            1 - existence_score, index=semseg_metadata.stuff_classes
        )
        for category_name, score in s_missing.items():
            if category_name == "BACKGROUND":
                continue

            row_results.append(
                {
                    "file_name": file_name,
                    "fdi": category_name.split("_")[-1],
                    "finding": "MISSING",
                    "score": score,
                }
            )

        #

        df_nontooth["score_per_tooth"] = None
        for index, row in df_nontooth.iterrows():
            # q_ij <= mean(prob over mask)
            share_per_tooth = calculate_mean_prob(
                row["mask"], bbox=row["bbox"], prob=prob, ignore_background=True
            )
            # p_ij <= 1 - (1 - p_j) ** q_ij
            score_per_tooth = 1 - np.power(1 - row["score"], share_per_tooth)

            df_nontooth.at[index, "score_per_tooth"] = score_per_tooth

        for finding in [
            "IMPLANT",
            "ROOT_REMNANTS",
            "CROWN_BRIDGE",
            "FILLING",
            "ENDO",
            "CARIES",
            "PERIAPICAL_RADIOLUCENT",
        ]:
            df_finding = df_nontooth.loc[df_nontooth["category_name"] == finding]
            if len(df_finding) == 0:
                continue

            p_tooth = np.stack(df_finding["score_per_tooth"], axis=0)
            finding_score: np.ndarray = 1 - np.prod(1 - p_tooth, axis=0)

            s_finding: pd.Series = pd.Series(
                finding_score, index=semseg_metadata.stuff_classes
            )
            for category_name, score in s_finding.items():
                if category_name == "BACKGROUND":
                    continue

                # for CROWN_BRIDGE, we do not assume tooth missing-ness
                # for IMPLANT, we need the tooth to be missing
                # for the rest, we need the tooth to be present
                match finding:
                    case "CROWN_BRIDGE":
                        pass

                    case "IMPLANT":
                        score = score * s_missing[category_name]

                    case _:
                        score = score * (1 - s_missing[category_name])

                row_results.append(
                    {
                        "file_name": file_name,
                        "fdi": category_name.split("_")[-1],
                        "finding": finding,
                        "score": score,
                    }
                )

        instances = parse_obj_as(
            list[InstanceDetectionPredictionInstance], df.to_dict(orient="records")
        )
        output_predictions.append(
            InstanceDetectionPrediction(
                image_id=prediction.image_id, instances=instances
            )
        )

    Path(FLAGS.result_dir).mkdir(parents=True, exist_ok=True)

    InstanceDetectionPredictionList.to_detectron2_detection_pth(
        output_predictions, path=Path(FLAGS.result_dir, FLAGS.output_prediction_name)
    )

    df_result: pd.DataFrame = pd.DataFrame(row_results).sort_values(
        ["file_name", "fdi", "finding"], ascending=True
    )
    df_result = df_result.drop_duplicates(
        subset=["file_name", "fdi", "finding"], keep="first"
    )
    df_result.to_csv(Path(FLAGS.result_dir, FLAGS.csv_name), index=False)


if __name__ == "__main__":
    app.run(main)
