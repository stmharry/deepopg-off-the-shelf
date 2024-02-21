import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rich.progress
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
    InstanceDetectionPredictionList,
)
from app.instance_detection.types import InstanceDetectionV1Category as Category
from app.masks import Mask
from app.semantic_segmentation.datasets import SemanticSegmentation
from app.utils import calculate_iom_bbox, calculate_iom_mask
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
flags.DEFINE_float("min_iom", 0.3, "Intersection over minimum threshold.")
flags.DEFINE_boolean("save_predictions", False, "Save predictions.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")

FLAGS = flags.FLAGS


def basic_filter(
    df: pd.DataFrame,
    min_score: float,
    min_area: float,
) -> pd.Series:
    keep: pd.Series = (df["area"] > min_area) & (df["score"] > min_score)

    logging.info(
        f"Filtering by area and score, reducing instances from {len(df)} -> {keep.sum()}"
    )
    return keep


def non_maximum_suppress(
    df: pd.DataFrame,
    iom_threshold: float,
) -> pd.Series:
    df = df.sort_values("score", ascending=False)
    num_instances: int = len(df)

    keep: pd.Series = pd.Series(True, index=df.index)
    for i in range(num_instances):
        row_i = df.iloc[i]

        if not keep.iloc[i]:
            continue

        for j in range(i + 1, num_instances):
            row_j = df.iloc[j]

            iom_bbox: float = calculate_iom_bbox(row_i["bbox"], row_j["bbox"])
            if iom_bbox == 0:
                continue

            iom_mask: float = calculate_iom_mask(row_i["mask"], row_j["mask"])
            if iom_mask > iom_threshold:
                keep.iloc[j] = False

    logging.info(
        f"Non-maximum suppression, reducing instances from {len(df)} -> {keep.sum()}"
    )
    return keep


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


def process_data(
    data: InstanceDetectionData,
    file_name: str,
    prediction: InstanceDetectionPrediction,
    metadata: Metadata,
    semseg_metadata: Metadata,
) -> tuple[InstanceDetectionPrediction | None, pd.DataFrame | None]:
    logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

    df: pd.DataFrame = pd.DataFrame.from_records(
        [instance.dict() for instance in prediction.instances]
    )
    if len(df) == 0:
        return None, None

    logging.info(f"Original instance count: {len(df)}.")

    df["mask"] = df["segmentation"].map(
        lambda segmentation: Mask.from_obj(
            segmentation, width=data.width, height=data.height
        ).bitmask
    )
    df["area"] = df["mask"].map(np.sum)

    keep: pd.Series = basic_filter(
        df, min_score=FLAGS.min_score, min_area=FLAGS.min_area
    )
    df = df.loc[keep]
    if len(df) == 0:
        return None, None

    df["category_name"] = df["category_id"].map(metadata.thing_classes.__getitem__)
    df["score_per_tooth"] = None

    npz_path: Path = Path(
        FLAGS.semseg_result_dir, "inference", f"{data.file_name.stem}.npz"
    )
    with np.load(npz_path) as npz:
        # prob.shape == (num_classes, height, width)
        prob: np.ndarray = npz["prob"]

    prob = prob.transpose(1, 2, 0)
    prob = prob.astype(np.float32) / 65535

    df_findings: list[pd.DataFrame] = []
    row_results: list[dict[str, Any]] = []

    s_missing: pd.Series | None = None
    for finding in Category:
        is_finding: pd.Series
        match finding:
            case Category.MISSING:
                # for tooth objects, additional nms is applied because in instance detection
                # we treat each fdi as a separate class

                is_tooth: pd.Series = df["category_name"].str.startswith("TOOTH")
                keep: pd.Series = non_maximum_suppress(
                    df.loc[is_tooth], iom_threshold=FLAGS.min_iom
                )
                is_finding = is_tooth & keep

            case _:
                is_finding = df["category_name"].eq(finding)  # type: ignore

        finding_score: np.ndarray
        if is_finding.sum() > 0:
            mask: np.ndarray = np.stack(df.loc[is_finding, "mask"], axis=-1)
            discount_factor: np.ndarray = np.sum(mask, axis=-1, keepdims=True)
            prob_discounted: np.ndarray = 1 - np.power(1 - prob, 1 / discount_factor)

            for index, row in df.loc[is_finding].iterrows():
                share_per_tooth: np.ndarray = calculate_mean_prob(
                    row["mask"],
                    bbox=row["bbox"],
                    prob=prob_discounted,
                    ignore_background=True,
                )

                score_per_tooth: np.ndarray
                match finding:
                    case Category.MISSING:
                        # we do not use objectiveness score for teeth
                        score_per_tooth = share_per_tooth

                    case _:
                        score_per_tooth = 1 - np.power(
                            1 - row["score"], share_per_tooth
                        )

                df.at[index, "score_per_tooth"] = score_per_tooth

            score_per_tooth = np.stack(df.loc[is_finding, "score_per_tooth"], axis=0)
            finding_score = 1 - np.prod(1 - score_per_tooth, axis=0)

        else:
            finding_score = np.zeros((len(semseg_metadata.stuff_classes),))

        s_finding: pd.Series = pd.Series(
            finding_score, index=semseg_metadata.stuff_classes
        )
        match finding:
            case Category.MISSING:
                s_finding = 1 - s_finding
                s_missing = s_finding

            case Category.CROWN_BRIDGE:
                # for CROWN_BRIDGE, we do not assume tooth missing-ness
                pass

            case Category.IMPLANT:
                # for IMPLANT, we need the tooth to be missing
                assert s_missing is not None
                s_finding = s_finding * s_missing

            case _:
                # for the rest, we need the tooth to be present
                assert s_missing is not None
                s_finding = s_finding * (1 - s_missing)

        for category_name, score in s_finding.items():
            if category_name == "BACKGROUND":
                continue

            assert isinstance(category_name, str)

            row_results.append(
                {
                    "file_name": file_name,
                    "fdi": category_name.split("_")[-1],
                    "finding": finding,
                    "score": score,
                }
            )

        df_findings.append(df.loc[is_finding])

    output_prediction: InstanceDetectionPrediction | None = None
    if FLAGS.save_predictions:
        df_finding = pd.concat(df_findings, axis=0)
        instances: list[InstanceDetectionPredictionInstance] = parse_obj_as(
            list[InstanceDetectionPredictionInstance],
            df_finding.to_dict(orient="records"),
        )
        output_prediction = InstanceDetectionPrediction(
            image_id=prediction.image_id, instances=instances
        )

    df_result: pd.DataFrame = pd.DataFrame(row_results)

    return output_prediction, df_result


def _process_data(
    args: tuple[
        InstanceDetectionData,
        str,
        InstanceDetectionPrediction,
        Metadata,
        Metadata,
    ],
) -> tuple[InstanceDetectionPrediction | None, pd.DataFrame | None]:
    warnings.simplefilter("ignore")
    logging.set_verbosity(logging.WARNING)

    return process_data(*args)


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

    # semantic segmentation

    semseg_data_driver: SemanticSegmentation | None = (
        SemanticSegmentation.register_by_name(
            dataset_name=FLAGS.semseg_dataset_name, root_dir=FLAGS.data_dir
        )
    )
    if semseg_data_driver is None:
        raise ValueError(f"Unknown dataset name: {FLAGS.semseg_dataset_name}")

    semseg_metadata: Metadata = MetadataCatalog.get(FLAGS.semseg_dataset_name)

    #

    with mp.Pool(processes=FLAGS.num_workers) as pool:
        tasks: list[tuple] = [
            (
                data,
                Path(data.file_name).relative_to(data_driver.image_dir).stem,
                id_to_prediction[data.image_id],
                metadata,
                semseg_metadata,
            )
            for data in dataset
            if data.image_id in id_to_prediction
        ]
        results = pool.imap_unordered(_process_data, tasks)

        output_predictions: list[InstanceDetectionPrediction] = []
        df_results: list[pd.DataFrame] = []
        for result in rich.progress.track(results, total=len(tasks)):
            _output_prediction: InstanceDetectionPrediction | None
            _df_result: pd.DataFrame | None
            _output_prediction, _df_result = result

            if _output_prediction is not None:
                output_predictions.append(_output_prediction)

            if _df_result is not None:
                df_results.append(_df_result)

    Path(FLAGS.result_dir).mkdir(parents=True, exist_ok=True)

    if FLAGS.save_predictions:
        InstanceDetectionPredictionList.to_detectron2_detection_pth(
            output_predictions,
            path=Path(FLAGS.result_dir, FLAGS.output_prediction_name),
        )

    df_result: pd.DataFrame = pd.concat(df_results, axis=0).sort_values(
        ["file_name", "fdi", "finding"], ascending=True
    )
    df_result = df_result.drop_duplicates(
        subset=["file_name", "fdi", "finding"], keep="first"
    )
    df_result.to_csv(Path(FLAGS.result_dir, FLAGS.csv_name), index=False)


if __name__ == "__main__":
    app.run(main)
