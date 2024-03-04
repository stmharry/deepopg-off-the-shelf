import contextlib
import enum
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.coco import ID
from app.instance_detection import (
    InstanceDetection,
    InstanceDetectionData,
    InstanceDetectionFactory,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
    InstanceDetectionPredictionList,
)
from app.instance_detection import InstanceDetectionV1Category as Category
from app.masks import Mask
from app.semantic_segmentation import SemanticSegmentation, SemanticSegmentationData
from app.tasks import Task, map_task
from app.utils import calculate_iom_bbox, calculate_iom_mask
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog


class ScoringMethod(enum.Flag):
    _USE_SHARE_WITH_BACKGROUND = enum.auto()
    _USE_SHARE_WITHOUT_BACKGROUND = enum.auto()

    _USE_SCORE_CONSTANT_ONE = enum.auto()
    _USE_SCORE_MULTIPLICATION = enum.auto()
    _USE_SCORE_DECOMPOSITION = enum.auto()

    _USE_MISSING_MULTIPLICATION = enum.auto()
    _NOUSE_MISSING_MULTIPLICATION = enum.auto()

    SHARE_BG = (
        _USE_SCORE_CONSTANT_ONE
        | _USE_SHARE_WITH_BACKGROUND
        | _USE_MISSING_MULTIPLICATION
    )
    SHARE_NOBG = (
        _USE_SCORE_CONSTANT_ONE
        | _USE_SHARE_WITHOUT_BACKGROUND
        | _USE_MISSING_MULTIPLICATION
    )
    SCORE_MUL_SHARE_BG = (
        _USE_SCORE_MULTIPLICATION
        | _USE_SHARE_WITH_BACKGROUND
        | _USE_MISSING_MULTIPLICATION
    )
    SCORE_MUL_SHARE_NOBG = (
        _USE_SCORE_MULTIPLICATION
        | _USE_SHARE_WITHOUT_BACKGROUND
        | _USE_MISSING_MULTIPLICATION
    )
    SCORE_DECOMP_USING_SHARE_BG = (
        _USE_SCORE_DECOMPOSITION
        | _USE_SHARE_WITH_BACKGROUND
        | _USE_MISSING_MULTIPLICATION
    )
    SCORE_DECOMP_USING_SHARE_NOBG = (
        _USE_SCORE_DECOMPOSITION
        | _USE_SHARE_WITHOUT_BACKGROUND
        | _USE_MISSING_MULTIPLICATION
    )
    SHARE_NOBG_NOMUL_MISSING = (
        _USE_SCORE_CONSTANT_ONE
        | _USE_SHARE_WITHOUT_BACKGROUND
        | _NOUSE_MISSING_MULTIPLICATION
    )

    @classmethod
    def members(cls) -> dict[str, "ScoringMethod"]:
        return {
            name: member
            for name, member in cls.__members__.items()
            if not name.startswith("_")
        }


flags.DEFINE_string("data_dir", "./data", "Data directory.")
flags.DEFINE_string("result_dir", "./results", "Result directory.")
flags.DEFINE_enum(
    "dataset_name", "pano", InstanceDetection.available_dataset_names(), "Dataset name."
)
flags.DEFINE_string(
    "prediction", "instances_predictions.pth", "Input prediction file name."
)
flags.DEFINE_bool(
    "use_gt_as_prediction",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground"
    " truth finding summary but only ground truth segmentation.",
)
flags.DEFINE_string(
    "semseg_result_dir", None, "Semantic segmentation result directory."
)
flags.DEFINE_enum(
    "semseg_dataset_name",
    "pano_semseg_v4",
    SemanticSegmentation.available_dataset_names(),
    "Semantic segmentation dataset name.",
)
flags.DEFINE_bool(
    "use_semseg_gt_as_prob",
    False,
    "Set to true to perform command on ground truth. Useful when we do not have ground"
    " truth finding summary but only ground truth segmentation.",
)
flags.DEFINE_string(
    "output_prediction",
    "instances_predictions.postprocessed.pth",
    "Output prediction file name.",
)
flags.DEFINE_string("output_csv", "result.csv", "Output result file name.")
flags.DEFINE_float("min_score", 0.01, "Confidence score threshold.")
flags.DEFINE_float("min_area", 0, "Object area threshold.")
flags.DEFINE_float("min_iom", 0.3, "Intersection over minimum threshold.")
flags.DEFINE_enum(
    "missing_scoring_method",
    ScoringMethod.SHARE_NOBG.name,
    ScoringMethod.members(),
    "Scoring method for missing finding.",
)
flags.DEFINE_enum(
    "finding_scoring_method",
    ScoringMethod.SCORE_DECOMP_USING_SHARE_NOBG.name,
    ScoringMethod.members(),
    "Scoring method for findings other than missing.",
)
flags.DEFINE_boolean("save_predictions", False, "Save predictions.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
FLAGS = flags.FLAGS


def non_maximum_suppress(
    df: pd.DataFrame,
    iom_threshold: float,
) -> pd.Series:
    df = df.sort_values("score", ascending=False)
    num_instances: int = len(df)

    keep: pd.Series = pd.Series(True, index=df.index)
    if iom_threshold == 1.0:
        return keep

    for i in range(num_instances):
        row_i = df.iloc[i]

        if not keep.iloc[i]:
            continue

        for j in range(i + 1, num_instances):
            row_j = df.iloc[j]

            iom_bbox: float = calculate_iom_bbox(row_i["bbox"], row_j["bbox"])
            if iom_bbox == 0:
                continue

            iom_mask: float = calculate_iom_mask(
                row_i["mask"], row_j["mask"], bbox1=row_i["bbox"], bbox2=row_j["bbox"]
            )
            if iom_mask > iom_threshold:
                keep.iloc[j] = False

    logging.info(
        f"Non-maximum suppression, reducing instances from {len(df)} -> {keep.sum()}"
    )
    return keep


def calculate_mean_prob(
    mask: np.ndarray,  # (H, W)
    prob_uint16: np.ndarray,  # (C, H, W)
    prob_discount_uint8: np.ndarray | None = None,  # (H, W)
    bbox: np.ndarray | None = None,
) -> np.ndarray:
    if bbox is not None:
        x, y, w, h = bbox
        slices: tuple[slice, slice] = (slice(y, y + h + 1), slice(x, x + w + 1))

        mask = mask[slices]
        prob_uint16 = prob_uint16[(..., *slices)]

        if prob_discount_uint8 is not None:
            prob_discount_uint8 = prob_discount_uint8[slices]

    if mask.sum() == 0:
        return np.zeros(prob_uint16.shape[0], dtype=np.float32)

    prob_uint16 = prob_uint16[..., mask]
    if prob_discount_uint8 is not None:
        prob_discount_uint8 = prob_discount_uint8[mask]

    prob_float32: np.ndarray = prob_uint16.astype(np.float32) / 65535
    if prob_discount_uint8 is not None:
        prob_discount_float32: np.ndarray = prob_discount_uint8.astype(np.float32)
        prob_float32 = 1 - np.power(1 - prob_float32, 1 / prob_discount_float32)

    return prob_float32.mean(axis=-1)


def calculate_score(
    total_score: float,
    share_including_bg: np.ndarray,
    scoring_method: ScoringMethod,
) -> np.ndarray:
    share: np.ndarray
    if scoring_method & ScoringMethod._USE_SHARE_WITH_BACKGROUND:
        share = share_including_bg

    elif scoring_method & ScoringMethod._USE_SHARE_WITHOUT_BACKGROUND:
        share = np.r_["-1,1,0", 0, share_including_bg[1:]]
        share_sum: float = share.sum()
        if share_sum != 0:
            share /= share.sum()

    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    score: np.ndarray
    if scoring_method & ScoringMethod._USE_SCORE_CONSTANT_ONE:
        score = share

    elif scoring_method & ScoringMethod._USE_SCORE_MULTIPLICATION:
        score = total_score * share

    elif scoring_method & ScoringMethod._USE_SCORE_DECOMPOSITION:
        score = 1 - np.power(1 - total_score, share)

    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    return score


def process_data(
    data: InstanceDetectionData,
    file_name: str,
    prediction: InstanceDetectionPrediction,
    metadata: Metadata,
    semseg_metadata: Metadata,
    semseg_data: SemanticSegmentationData | None,
    semseg_result_dir: Path | None,
    min_score: float,
    min_area: float,
    min_iom: float,
    missing_scoring_method: ScoringMethod,
    finding_scoring_method: ScoringMethod,
    save_predictions: bool,
) -> tuple[InstanceDetectionPrediction | None, pd.DataFrame] | None:
    logging.info(f"Processing {data.file_name} with image id {data.image_id}.")

    df: pd.DataFrame = pd.DataFrame.from_records(
        [instance.dict() for instance in prediction.instances]
    )
    logging.info(f"Original instance count: {len(df)}.")
    if len(df) == 0:
        return None

    df = df.loc[df["score"] > min_score]
    if len(df) == 0:
        return None

    df["mask"] = df["segmentation"].map(
        lambda segmentation: Mask.from_obj(
            segmentation, width=data.width, height=data.height
        ).bitmask
    )
    df["area"] = df["mask"].map(np.sum)

    df = df.loc[df["area"] > min_area]
    if len(df) == 0:
        return None

    df["category_name"] = df["category_id"].map(metadata.thing_classes.__getitem__)
    df["score_per_tooth"] = None

    prob_uint16: np.ndarray
    if semseg_data is not None:
        assert semseg_data.sem_seg_file_name is not None
        label_uint8: np.ndarray = iio.imread(semseg_data.sem_seg_file_name)

        prob_bool: np.ndarray = (
            label_uint8[None, ...]
            == np.arange(len(semseg_metadata.stuff_classes))[:, None, None]
        )
        prob_uint16 = np.where(prob_bool, np.uint16(65535), np.uint16(0))

    elif semseg_result_dir is not None:
        npz_path: Path = Path(
            semseg_result_dir, "inference", f"{data.file_name.stem}.npz"
        )
        with np.load(npz_path) as npz:
            prob_uint16 = npz["prob"]  # (C, H, W)

    else:
        raise ValueError(
            "Either `semseg_data` or `semseg_result_dir` must be provided."
        )

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
                    df.loc[is_tooth], iom_threshold=min_iom
                )
                is_finding = is_tooth & keep

            case _:
                is_finding = df["category_name"].eq(finding)  # type: ignore

        finding_score: np.ndarray
        if is_finding.sum() > 0:
            masks: list[np.ndarray] = [
                mask.astype(np.uint8) for mask in df.loc[is_finding, "mask"].tolist()
            ]
            prob_discount_uint8: np.ndarray = sum(masks[1:], start=masks[0])

            for index, row in df.loc[is_finding].iterrows():
                scoring_method: ScoringMethod
                match finding:
                    case Category.MISSING:
                        scoring_method = missing_scoring_method

                    case _:
                        scoring_method = finding_scoring_method

                share_including_bg: np.ndarray = calculate_mean_prob(
                    row["mask"],
                    bbox=row["bbox"],
                    prob_uint16=prob_uint16,
                    prob_discount_uint8=prob_discount_uint8,
                )
                score_per_tooth: np.ndarray = calculate_score(
                    total_score=row["score"],
                    share_including_bg=share_including_bg,
                    scoring_method=scoring_method,
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
                if finding_scoring_method & ScoringMethod._USE_MISSING_MULTIPLICATION:
                    assert s_missing is not None
                    # for IMPLANT, we need the tooth to be missing
                    s_finding = s_finding * s_missing

                else:
                    # no modification to the score
                    pass

            case _:
                if finding_scoring_method & ScoringMethod._USE_MISSING_MULTIPLICATION:
                    # for the rest, we need the tooth to be present
                    assert s_missing is not None
                    s_finding = s_finding * (1 - s_missing)

                else:
                    # no modification to the score
                    pass

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
    if save_predictions:
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


def main(_):
    # instance detection

    data_driver: InstanceDetection | None = InstanceDetectionFactory.register_by_name(
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
            Path(FLAGS.result_dir, FLAGS.prediction),
            image_ids=set(data.image_id for data in dataset),
        )

    id_to_prediction: dict[ID, InstanceDetectionPrediction] = {
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

    id_to_semseg_data: dict[ID, SemanticSegmentationData] = {}
    if FLAGS.use_semseg_gt_as_prob:
        semseg_dataset: list[SemanticSegmentationData] = parse_obj_as(
            list[SemanticSegmentationData],
            DatasetCatalog.get(FLAGS.semseg_dataset_name),
        )
        id_to_semseg_data = {data.image_id: data for data in semseg_dataset}

    #

    output_predictions: list[InstanceDetectionPrediction] = []
    df_results: list[pd.DataFrame] = []
    with contextlib.ExitStack() as stack:
        tasks: list[Task] = [
            Task(
                fn=process_data,
                kwargs={
                    "data": data,
                    "file_name": (
                        Path(data.file_name).relative_to(data_driver.image_dir).stem
                    ),
                    "prediction": id_to_prediction[data.image_id],
                    "metadata": metadata,
                    "semseg_data": id_to_semseg_data.get(data.image_id),
                    "semseg_metadata": semseg_metadata,
                    "semseg_result_dir": (
                        Path(FLAGS.semseg_result_dir)
                        if FLAGS.semseg_result_dir
                        else None
                    ),
                    "min_score": FLAGS.min_score,
                    "min_area": FLAGS.min_area,
                    "min_iom": FLAGS.min_iom,
                    "missing_scoring_method": ScoringMethod[
                        FLAGS.missing_scoring_method
                    ],
                    "finding_scoring_method": ScoringMethod[
                        FLAGS.finding_scoring_method
                    ],
                    "save_predictions": FLAGS.save_predictions,
                },
            )
            for data in dataset
            if data.image_id in id_to_prediction
        ]

        for result in map_task(tasks, stack=stack, num_workers=FLAGS.num_workers):
            if result is None:
                continue

            _output_prediction: InstanceDetectionPrediction | None
            _df_result: pd.DataFrame
            _output_prediction, _df_result = result

            if _output_prediction is not None:
                output_predictions.append(_output_prediction)

            df_results.append(_df_result)

    Path(FLAGS.result_dir).mkdir(parents=True, exist_ok=True)

    if FLAGS.save_predictions:
        detection_path: Path = Path(FLAGS.result_dir, FLAGS.output_prediction)
        logging.info(f"Saving predictions to {detection_path}")

        InstanceDetectionPredictionList.to_detectron2_detection_pth(
            output_predictions, path=detection_path
        )

    df_result: pd.DataFrame = pd.concat(df_results, axis=0).sort_values(
        ["file_name", "fdi", "finding"], ascending=True
    )
    df_result = df_result.drop_duplicates(
        subset=["file_name", "fdi", "finding"], keep="first"
    )

    csv_path: Path = Path(FLAGS.result_dir, FLAGS.output_csv)
    logging.info(f"Saving results to {csv_path}")
    df_result.to_csv(csv_path, index=False)


if __name__ == "__main__":
    app.run(main)
