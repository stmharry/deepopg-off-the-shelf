from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import scipy.interpolate
import scipy.ndimage
from absl import app, flags, logging
from pydantic import parse_obj_as

from app.instance_detection.datasets import InstanceDetection
from app.instance_detection.schemas import (
    InstanceDetectionData,
    InstanceDetectionPrediction,
    InstanceDetectionPredictionInstance,
)
from app.utils import (
    Mask,
    calculate_iom_bbox,
    calculate_iom_mask,
    instance_detection_data_to_prediction,
    load_predictions,
    save_predictions,
)
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

flags.DEFINE_string("data_dir", None, "Data directory.")
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string(
    "prediction_name", "instances_predictions.pth", "Input prediction file name."
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
flags.DEFINE_string("output_csv_name", "result.csv", "Output result file name.")
flags.DEFINE_float("min_score", 0.01, "Confidence score threshold.")
flags.DEFINE_integer(
    "tooth_distance",
    150,
    "Implant has to be within this distance to be considered valid.",
)

FLAGS = flags.FLAGS


def do_assignment(
    reward: np.ndarray,
    quadratic_penalty: np.ndarray,
    penalty_group_ids: list[int | None] | None = None,
    unique_group_ids: list[int | None] | None = None,
    assignment_penalty: float = 0.01,
    epsilon: float = 1e-3,
) -> np.ndarray:
    assignment: np.ndarray

    num: int = len(reward)
    if num == 0:
        assignment = np.zeros(0, dtype=np.bool_)
        return assignment

    if penalty_group_ids is None:
        penalty_group_ids = [None] * num

    if unique_group_ids is None:
        unique_group_ids = [None] * num

    assert quadratic_penalty.shape == (num, num)
    assert penalty_group_ids is not None
    assert unique_group_ids is not None

    p: dict[int, float] = {n: reward[n] for n in range(num)}
    q: dict[tuple[int, int], float] = {
        (n1, n2): quadratic_penalty[n1, n2] for n1 in range(num) for n2 in range(num)
    }

    model: pyo.ConcreteModel = pyo.ConcreteModel("QuadraticAssignment")
    model.N = pyo.RangeSet(0, num - 1)
    model.P = pyo.Param(model.N, initialize=p)
    model.Q = pyo.Param(model.N, model.N, initialize=q)

    model.x = pyo.Var(model.N, domain=pyo.Binary)

    objectives: list[pyo.Expression] = []
    for n1 in model.N:
        objectives.append(model.P[n1] * model.x[n1])

        for n2 in model.N:
            # only deal with upper triangular matrix
            if n1 > n2:
                continue

            penalty_coefficient: float = 0.0
            if n1 == n2:
                penalty_coefficient = assignment_penalty
            elif (
                penalty_group_ids[n1] is not None
                and penalty_group_ids[n2] is not None
                and penalty_group_ids[n1] == penalty_group_ids[n2]
            ):
                penalty_coefficient = model.Q[n1, n2]

            if penalty_coefficient == 0.0:
                continue

            objectives.append(-penalty_coefficient * model.x[n1] * model.x[n2])

    constraints: list[pyo.Expression] = []
    if unique_group_ids is not None:
        for unique_group_id in list(set(unique_group_ids)):
            if unique_group_id is None:
                continue

            constraints.append(
                sum(
                    model.x[n]
                    for n in model.N
                    if unique_group_ids[n] == unique_group_id
                )
                <= 1
            )

    model.obj = pyo.Objective(expr=sum(objectives), sense=pyo.maximize)
    if len(constraints) > 0:
        model.constraints = pyo.ConstraintList()
        for constraint in constraints:
            model.constraints.add(constraint)

    solver = pyo.SolverFactory("ipopt")
    solver.solve(model)

    assignment = np.zeros(num, dtype=np.bool_)
    has_invalid_value: bool = False
    for n in model.N:
        value: float = model.x[n].value

        if abs(value - 1.0) < epsilon:
            assignment[n] = True
        elif abs(value - 0.0) < epsilon:
            assignment[n] = False
        else:
            has_invalid_value = True
            assignment[n] = False

    if has_invalid_value:
        logging.warning("Invalid value detected in assignment.")

    return assignment


def main(_):
    logging.set_verbosity(logging.INFO)

    data_driver: InstanceDetection = InstanceDetection.register_by_name(
        dataset_name=FLAGS.dataset_name, root_dir=FLAGS.data_dir
    )
    dataset: list[InstanceDetectionData] = parse_obj_as(
        list[InstanceDetectionData], DatasetCatalog.get(FLAGS.dataset_name)
    )
    metadata: Metadata = MetadataCatalog.get(FLAGS.dataset_name)

    predictions: list[InstanceDetectionPrediction]
    if FLAGS.use_gt_as_prediction:
        predictions = [
            instance_detection_data_to_prediction(instance_detection_data=data)
            for data in dataset
        ]

    else:
        predictions = load_predictions(
            prediction_path=Path(FLAGS.result_dir, FLAGS.prediction_name)
        )

    id_to_prediction: dict[str | int, InstanceDetectionPrediction] = {
        prediction.image_id: prediction for prediction in predictions
    }

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

        df["area"] = df["segmentation"].apply(
            lambda segmentation: Mask.from_obj(
                segmentation, width=data.width, height=data.height
            ).area
        )
        df = df.loc[df.area > 0]
        logging.info(f"Found {len(df)} instances.")

        if len(df) == 0:
            continue

        # we have to find a way to "assign" a score for `MISSING`, so we find the max raw
        # detection score in the original instance list and subtract it from one
        missingness: pd.Series = np.maximum(  # type: ignore
            0, 1 - df.groupby("category_id").score.max()
        )

        df = df.loc[df.score > FLAGS.min_score]
        logging.info(f"Found {len(df)} instances with score > {FLAGS.min_score}.")

        if len(df) == 0:
            continue

        df["category_name"] = df["category_id"].map(metadata.thing_classes.__getitem__)
        df["is_tooth"] = df["category_name"].str.startswith("TOOTH")
        df["fdi"] = pd.Series([None] * len(df), index=df.index).mask(
            df["is_tooth"], other=df["category_name"].str.split("_").str[-1]
        )
        df["mask"] = df["segmentation"].apply(
            lambda segmentation: Mask.from_obj(
                segmentation, width=data.width, height=data.height
            ).bitmask
        )
        num_instances: int = len(df)

        # instance filtering

        df["penalty_group_id"] = (
            pd.Series(0, index=df.index, dtype=int)
            .mask(df["is_tooth"], other=1)
            .mask(df["category_name"] == "ROOT_REMNANTS", other=2)
        )
        df["unique_group_id"] = pd.Series(
            [None] * len(df), index=df.index, dtype=object
        ).mask(df["is_tooth"], other=df["category_id"])

        iom: np.ndarray = np.eye(num_instances)
        iom_bbox: float
        iom_mask: float
        for i in range(num_instances):
            for j in range(num_instances):
                if i >= j:
                    continue

                iom_bbox = calculate_iom_bbox(df.iloc[i]["bbox"], df.iloc[j]["bbox"])
                if iom_bbox == 0:
                    continue

                iom_mask = calculate_iom_mask(df.iloc[i]["mask"], df.iloc[j]["mask"])
                logging.debug(f"IoM of instances {i} and {j} is {iom_mask}.")

                iom[i, j] = iom[j, i] = iom_mask

        assignment: np.ndarray = do_assignment(
            reward=df["score"].to_numpy(),
            quadratic_penalty=iom,
            penalty_group_ids=df["penalty_group_id"].tolist(),
            unique_group_ids=df["unique_group_id"].tolist(),
            assignment_penalty=FLAGS.min_score / 2,
        )
        df = df.loc[assignment]
        logging.info(f"Found {len(df)} instances after quadratic assignment.")

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

            # for other findings
            else:
                distance_to_non_tooth_instance = cast(
                    np.ndarray,
                    scipy.ndimage.distance_transform_cdt(
                        1 - row_nontooth["mask"][all_instances_slice]
                    ),
                )

                dist = np.zeros((num_tooth,), dtype=np.float_)
                for i in range(num_tooth):
                    row_tooth = df_tooth.iloc[i]

                    dist[i] = scipy.ndimage.minimum(
                        distance_to_non_tooth_instance,
                        labels=row_tooth["mask"][all_instances_slice],
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

    save_predictions(
        predictions=output_predictions,
        prediction_path=Path(FLAGS.result_dir, FLAGS.output_prediction_name),
    )

    df_result: pd.DataFrame = pd.DataFrame(row_results).sort_values(
        ["file_name", "fdi", "finding"], ascending=True
    )
    df_result = df_result.drop_duplicates(
        subset=["file_name", "fdi", "finding"], keep="first"
    )
    df_result.to_csv(Path(FLAGS.result_dir, FLAGS.output_csv_name), index=False)


if __name__ == "__main__":
    app.run(main)
