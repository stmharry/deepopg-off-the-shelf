from pathlib import Path

import pandas as pd
from absl import app, flags, logging
from sklearn.metrics import auc, roc_curve

# common arguments
flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("output_csv_name", "result.csv", "Output result file name.")
flags.DEFINE_string("golden_dir", None, "Golden label directory.")
flags.DEFINE_string("golden_csv_name", None, "Golden label file name.")
# ROC curve
flags.DEFINE_bool("do_roc", False, "Whether to do ROC curve.")
# accuracy of the number of implant
flags.DEFINE_bool("do_acc", False, "Whether to do # implant acc.")
FLAGS = flags.FLAGS


def ROC(df_golden: pd.DataFrame, df_result: pd.DataFrame) -> None:
    list_finding = [
        "MISSING",
        "IMPLANT",
        "ROOT_REMNANTS",
        "CARIES",
        "ENDO",
        "CROWN_BRIDGE",
        "PERIAPICAL_RADIOLUCENT",
        "FILLING",
    ]
    file_list = pd.unique(df_golden["file_name"]).tolist()

    for finding_name in list_finding:
        df_golden_finding = df_golden[df_golden["finding"] == finding_name]
        df_result_finding = df_result[df_result["finding"] == finding_name]

        df_auc = pd.DataFrame()
        for i in file_list:
            file_golden = df_golden_finding[df_golden_finding["file_name"] == i]
            file_result = df_result_finding[df_result_finding["file_name"] == i]

            df_full_tooth: pd.DataFrame = pd.DataFrame(
                [
                    {"fdi": int(f"{quadrant}{tooth}")}
                    for quadrant in range(1, 5)
                    for tooth in range(1, 9)
                ]
            )

            df_full_tooth["golden"] = (
                df_full_tooth["fdi"].isin(file_golden["fdi"]).astype(int)
            )
            df_full_tooth = pd.merge(
                df_full_tooth,
                file_result[["fdi", "score"]],
                left_on="fdi",
                right_on="fdi",
                how="left",
            )
            df_full_tooth["score"].fillna(0, inplace=True)

            df_auc = pd.concat([df_auc, df_full_tooth], ignore_index=True)

        golden = df_auc["golden"]
        score = df_auc["score"]

        fpr, tpr, thresholds = roc_curve(golden, score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        print(finding_name)
        print(roc_auc)


def ACC(
    df_golden: pd.DataFrame,
    df_result: pd.DataFrame
) -> None:
    df_count = df_result[df_result["finding"] == "IMPLANT"].groupby("file_name").size().reset_index(name='num_implant')
    unique_names = df_result['file_name'].unique()
    missing_names = set(unique_names) - set(df_count['file_name'])
    missing_data = pd.DataFrame({'file_name': list(missing_names), 'num_implant': [0] * len(missing_names)})
    df_count = pd.concat([df_count, missing_data], ignore_index=True)

    df_golden["finding"] = df_golden["finding"] == "IMPLANT"
    df_golden = df_golden.groupby("file_name")["finding"].sum().reset_index()

    # names_to_remove = df_implant.loc[df_implant["exclude"] == True, "file_name"]
    # df_golden = df_golden[~df_golden["file_name"].isin(names_to_remove)]
    # df_count = df_count[~df_count["file_name"].isin(names_to_remove)]

    merged_df = df_golden.merge(df_count, on="file_name", how="left")

    acc = sum(merged_df["finding"] == merged_df["num_implant"]) / len(merged_df)
    print(acc)


def main(_):
    logging.set_verbosity(logging.INFO)
    df_golden = pd.read_csv(Path(FLAGS.golden_dir, FLAGS.golden_csv_name))
    df_result = pd.read_csv(Path(FLAGS.result_dir, FLAGS.output_csv_name))

    if FLAGS.do_roc:
        ROC(df_golden=df_golden, df_result=df_result)
    if FLAGS.do_acc:
        ACC(df_golden=df_golden, df_result=df_result)


if __name__ == "__main__":
    app.run(main)
