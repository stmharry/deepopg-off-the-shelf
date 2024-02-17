from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.ndimage
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
flags.DEFINE_integer(
    "tooth_distance",
    150,
    "Implant has to be within this distance to be considered valid.",
)

FLAGS = flags.FLAGS


def filter_by_area_and_score(
    df: pd.DataFrame,
    min_score: float,
    min_area: float,
) -> pd.DataFrame:
    _df = df.loc[(df["area"] > min_area) & (df["score"] > min_score)]

    logging.info(
        f"Filtering by area and score, reducing instances from {len(df)} -> {len(_df)}"
    )
    return _df


def non_maximum_suppression(
    df: pd.DataFrame,
    iom_threshold: float,
) -> pd.DataFrame:
    df = df.sort_values("score", ascending=False)
    df["nms_keep"] = True
    num_instances: int = len(df)

    for i in range(num_instances):
        row_i = df.iloc[i]

        if not row_i["nms_keep"]:
            continue

        for j in range(i + 1, num_instances):
            row_j = df.iloc[j]

            iom_bbox: float = calculate_iom_bbox(row_i["bbox"], row_j["bbox"])
            if iom_bbox == 0:
                continue

            iom_mask: float = calculate_iom_mask(row_i["mask"], row_j["mask"])
            if iom_mask > iom_threshold:
                df.at[row_j.name, "nms_keep"] = False

    _df = df.loc[df["nms_keep"]].drop(columns=["nms_keep"])
    logging.info(
        f"Non-maximum suppression, reducing instances from {len(df)} -> {len(_df)}"
    )
    return _df


def reassign_category_and_score_with_semseg(
    df: pd.DataFrame,
    npz_path: Path,
    metadata: Metadata,
    semseg_metadata: Metadata,
) -> pd.DataFrame:
    # when semseg dataset is passed, we replace the score for teeth with the overlap
    # between the instance and each tooth class from semseg, instead of using the
    # class score from instance detection

    with np.load(npz_path) as data:
        # prob.shape == (num_classes, height, width)
        prob: np.ndarray = data["prob"]

    prob = prob.transpose(1, 2, 0)
    prob = prob.astype(np.float32) / 65535

    df["prob"] = None
    for index, row in df.iterrows():
        row_prob = np.r_["-1,1,0", 0, prob[row["mask"]].mean(axis=0)[1:]]
        # renormalize, discarding the background class
        row_prob = row_prob / row_prob.sum()

        df.at[index, "prob"] = row_prob

        semseg_category_id: int = np.argmax(row_prob)
        score: float = row_prob[semseg_category_id]

        category_name = semseg_metadata.stuff_classes[semseg_category_id]
        category_id = metadata.thing_classes.index(category_name)

        if category_id == row["category_id"]:
            logging.info(
                f"Rescoring instance of category {row['category_name']} with score {row['score']:.4f} "
                f"to score {score:.4f}."
            )
        else:
            logging.info(
                f"Relabeling instance of category {row['category_name']} with score {row['score']:.4f} "
                f"to category {category_name} with score {score:.4f}."
            )

        df.at[index, "score"] = score
        df.at[index, "category_id"] = category_id

    df["category_name"] = df["category_id"].map(metadata.thing_classes.__getitem__)

    return df


def select_top_instance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    _df = df.sort_values("score", ascending=False).groupby("category_name").head(1)
    logging.info(
        f"Selecting top instance, reducing instances from {len(df)} -> {len(_df)}"
    )
    return _df


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

    # semantic segmentation, if specified

    semseg_data_driver: SemanticSegmentation | None = None
    semseg_metadata: Metadata | None = None
    if FLAGS.semseg_dataset_name is not None:
        semseg_data_driver = SemanticSegmentation.register_by_name(
            dataset_name=FLAGS.semseg_dataset_name, root_dir=FLAGS.data_dir
        )
        if semseg_data_driver is None:
            raise ValueError(f"Unknown dataset name: {FLAGS.semseg_dataset_name}")

        semseg_metadata = MetadataCatalog.get(FLAGS.semseg_dataset_name)

    output_predictions: list[InstanceDetectionPrediction] = []
    row_results: list[dict[str, Any]] = []
    for data in dataset:
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

        # basic filtering

        df["mask"] = df["segmentation"].map(
            lambda segmentation: Mask.from_obj(
                segmentation, width=data.width, height=data.height
            ).bitmask
        )
        df["area"] = df["mask"].map(np.sum)

        df = filter_by_area_and_score(df, min_score=FLAGS.min_score, min_area=0.0)

        if len(df) == 0:
            continue

        df["category_name"] = df["category_id"].map(metadata.thing_classes.__getitem__)
        df["is_tooth"] = df["category_name"].str.startswith("TOOTH")

        if FLAGS.semseg_dataset_name is not None:
            df_tooth = df.loc[df["is_tooth"]]
            df_nontooth = df.loc[~df["is_tooth"]]

            # non-maximum supression for tooth classes
            df_tooth = non_maximum_suppression(df_tooth, iom_threshold=0.7)
            df_tooth = reassign_category_and_score_with_semseg(
                df_tooth,
                npz_path=Path(
                    FLAGS.semseg_result_dir, "inference", f"{data.file_name.stem}.npz"
                ),
                metadata=metadata,
                semseg_metadata=semseg_metadata,
            )
            df_tooth = select_top_instance(df_tooth)

            df = pd.concat([df_tooth, df_nontooth], axis=0)

        df["fdi"] = pd.Series([None] * len(df), index=df.index).mask(
            df["is_tooth"], other=df["category_name"].str.split("_").str[-1]
        )

        instances = parse_obj_as(
            list[InstanceDetectionPredictionInstance], df.to_dict(orient="records")
        )
        output_predictions.append(
            InstanceDetectionPrediction(
                image_id=prediction.image_id, instances=instances
            )
        )

        # result csv

        all_instances_mask: np.ndarray = np.logical_or.reduce(
            df["mask"].tolist(), axis=0
        )
        all_instances_slice: tuple[slice, slice] = scipy.ndimage.find_objects(
            all_instances_mask, max_label=1
        )[0]

        df_tooth: pd.DataFrame = df.loc[df["is_tooth"]]
        df_nontooth: pd.DataFrame = df.loc[~df["is_tooth"]]

        num_tooth: int = len(df_tooth)
        num_nontooth: int = len(df_nontooth)

        row_tooth: pd.Series | None
        row_nontooth: pd.Series

        for j in range(num_nontooth):
            row_tooth = None
            row_nontooth = df_nontooth.iloc[j]

            distance_to_non_tooth_instance: np.ndarray
            dist: np.ndarray

            # for `IMPLANT`
            if row_nontooth["category_name"] == "IMPLANT":
                # For implant, we perform the following steps to find the closest tooth instance:

                ### 1. Instantiate a `DataFrame` of all teeth

                df_full_tooth: pd.DataFrame = pd.DataFrame(
                    [
                        {"fdi": f"{quadrant}{tooth}"}
                        for quadrant in range(1, 5)
                        for tooth in range(1, 9)
                    ]
                )

                ### 2. Pull in relevant info for exising teeth

                df_full_tooth = pd.merge(
                    df_full_tooth,
                    df_tooth[["fdi", "bbox"]],
                    on="fdi",
                    how="left",
                )
                df_full_tooth["exists"] = df_full_tooth["fdi"].isin(df_tooth["fdi"])

                ### 3. We put all teeth on a regular grid, and impute the missing bbox centers
                #
                #  The grid is defined as follows:
                #
                #  |--------------|--------------|-----|--------------|--------------|
                #  |     [18]     |     [17]     | ... |     [27]     |     [28]     |
                #  | (-7.5, +1.0) | (-6.5, +1.0) | ... | (+6.5, +1.0) | (+7.5, +1.0) |
                #  |--------------|--------------|-----|--------------|--------------|
                #  |     [48]     |     [47]     | ... |     [37]     |     [38]     |
                #  | (-7.5, -1.0) | (-6.5, -1.0) | ... | (+6.5, -1.0) | (+7.5, -1.0) |
                #  |--------------|--------------|-----|--------------|--------------|

                df_full_tooth["fdi_int"] = df_full_tooth["fdi"].astype(int)
                df_full_tooth["top"] = pd.Series.isin(
                    df_full_tooth["fdi_int"] // 10, [1, 2]
                )
                df_full_tooth["left"] = pd.Series.isin(
                    df_full_tooth["fdi_int"] // 10, [2, 3]
                )
                df_full_tooth["x"] = (df_full_tooth["fdi_int"] % 10 - 0.5) * np.where(
                    df_full_tooth["left"], 1, -1
                )
                df_full_tooth["y"] = np.where(df_full_tooth["top"], 1, -1)

                df_full_tooth.loc[
                    df_full_tooth["exists"], "bbox_x_center"
                ] = df_full_tooth.loc[df_full_tooth["exists"], "bbox"].map(
                    lambda bbox: bbox[0] + bbox[2] / 2
                )
                df_full_tooth.loc[
                    df_full_tooth["exists"], "bbox_y_center"
                ] = df_full_tooth.loc[df_full_tooth["exists"], "bbox"].map(
                    lambda bbox: bbox[1] + bbox[3] / 2
                )

                try:
                    interp = scipy.interpolate.RBFInterpolator(
                        y=df_full_tooth.loc[df_full_tooth["exists"], ["x", "y"]],
                        d=df_full_tooth.loc[
                            df_full_tooth["exists"], ["bbox_x_center", "bbox_y_center"]
                        ],
                        smoothing=1e0,
                        kernel="thin_plate_spline",
                    )
                except ValueError:
                    logging.warning(
                        f"ValueError encountered when interpolating bbox centers for {row_nontooth['fdi']}."
                    )
                    continue

                df_full_tooth[
                    ["bbox_x_center_interp", "bbox_y_center_interp"]
                ] = interp(
                    x=df_full_tooth[["x", "y"]],
                )

                ### 4. We compute the distance from the implant to the each tooth

                distance_to_non_tooth_instance = cast(
                    np.ndarray,
                    scipy.ndimage.distance_transform_cdt(
                        1 - row_nontooth["mask"][all_instances_slice]
                    ),
                )

                dist = np.zeros((len(df_full_tooth),), dtype=np.float_)
                for i in range(len(df_full_tooth)):
                    _row_tooth = df_full_tooth.iloc[i]

                    # calculate the indices on the distance map
                    y_index: int = min(
                        all_instances_slice[0].stop - all_instances_slice[0].start - 1,
                        max(
                            0,
                            int(
                                _row_tooth["bbox_y_center_interp"]
                                - all_instances_slice[0].start
                            ),
                        ),
                    )
                    x_index: int = min(
                        all_instances_slice[1].stop - all_instances_slice[1].start - 1,
                        max(
                            0,
                            int(
                                _row_tooth["bbox_x_center_interp"]
                                - all_instances_slice[1].start
                            ),
                        ),
                    )

                    dist[i] = distance_to_non_tooth_instance[y_index, x_index]

                df_full_tooth["dist"] = dist

                if not df_full_tooth["exists"].all():
                    idx: int = df_full_tooth["dist"].idxmin()

                    # if the finding distance to non_tooth is too far than filter
                    if dist[idx] < FLAGS.tooth_distance:
                        row_tooth = df_full_tooth.loc[idx]

            elif row_nontooth["category_name"] in [
                "ROOT_REMNANTS",
                "CROWN_BRIDGE",
                "FILLING",
                "ENDO",
                "CARIES",
            ]:
                # _iom = iom[df.index.get_loc(row_nontooth.name), df["is_tooth"]]
                #
                # if num_tooth and _iom.max() > 0.001:
                #     row_tooth = df_tooth.iloc[np.argmax(_iom)]

                iom: np.ndarray = np.zeros((num_tooth,), dtype=np.float_)
                for i in range(num_tooth):
                    _row_tooth = df_tooth.iloc[i]

                    iom_bbox = calculate_iom_bbox(
                        row_nontooth["bbox"], _row_tooth["bbox"]
                    )
                    if iom_bbox == 0:
                        continue

                    iom_mask = calculate_iom_mask(
                        row_nontooth["mask"],
                        _row_tooth["mask"],
                        bbox1=row_nontooth["bbox"],
                        bbox2=_row_tooth["bbox"],
                    )
                    logging.debug(f"iom_mask between {i} and {j} is {iom_mask}")

                    iom[i] = iom_mask

                if num_tooth:
                    row_tooth = df_tooth.iloc[np.argmax(iom)]

            if row_nontooth["category_name"] == "PERIAPICAL_RADIOLUCENT":
                distance_to_non_tooth_instance = cast(
                    np.ndarray,
                    scipy.ndimage.distance_transform_cdt(
                        1 - row_nontooth["mask"][all_instances_slice]
                    ),
                )

                dist = np.zeros((num_tooth,), dtype=np.float_)
                for i in range(num_tooth):
                    _row_tooth = df_tooth.iloc[i]

                    dist[i] = scipy.ndimage.minimum(
                        distance_to_non_tooth_instance,
                        labels=_row_tooth["mask"][all_instances_slice],
                    )

                if num_tooth:
                    row_tooth = df_tooth.iloc[np.argmin(dist)]

            if row_tooth is None:
                continue

            assert row_tooth is not None

            row_results.append(
                {
                    "file_name": file_name,
                    "fdi": int(row_tooth["fdi"]),
                    "finding": row_nontooth["category_name"],
                    "score": row_nontooth["score"],
                }
            )

        # we have to find a way to "assign" a score for `MISSING`, so we find the max raw
        # detection score in the original instance list and subtract it from one
        missingness: pd.Series = np.maximum(  # type: ignore
            0, 1 - df.groupby("category_id").score.max()
        )

        # for missing
        for category_id, category in enumerate(metadata.thing_classes):
            if not category.startswith("TOOTH"):
                continue

            row_results.append(
                {
                    "file_name": file_name,
                    "fdi": int(category.split("_")[1]),
                    "finding": "MISSING",
                    "score": missingness.get(category_id, 1.0),  # type: ignore
                }
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
