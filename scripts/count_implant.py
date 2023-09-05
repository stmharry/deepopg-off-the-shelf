import pandas as pd
from absl import flags
from pathlib import Path
import sys

flags.DEFINE_string("result_dir", None, "Result directory.")
flags.DEFINE_string("sheet_dir", None, "Sheet directory.")

FLAGS = flags.FLAGS

def count_implant(argv) -> None:
    flags.FLAGS(argv)

    df_implant = pd.read_csv(Path(FLAGS.sheet_dir, "PANO_implant.csv"))
    df_finding = pd.read_csv(Path(FLAGS.sheet_dir, "PANO_finding.csv"))
    df_result = pd.read_csv(Path(FLAGS.result_dir, "count_implant.csv"))

    df_finding["finding"] = df_finding["finding"] == "IMPLANT"
    df_golden = df_finding.groupby("file_name")["finding"].sum().reset_index()

    # remove exclude
    names_to_remove = df_implant.loc[df_implant["exclude"] == True, "file_name"]
    df_golden = df_golden[~df_golden["file_name"].isin(names_to_remove)]
    df_result = df_result[~df_result["file_name"].isin(names_to_remove)]

    merged_df = df_golden.merge(df_result, on="file_name", how="left")

    acc = sum(merged_df["finding"] == merged_df["num_implant"]) / len(merged_df)
    print("accuracy of implants:", acc)

    wrong = merged_df.loc[
        merged_df["finding"] != merged_df["num_implant"],
        ["file_name", "finding", "num_implant"],
    ]
    df_result: pd.DataFrame = pd.DataFrame(wrong)
    df_result.to_csv(Path(FLAGS.result_dir, "wrong.csv"), index=False)

if __name__ == "__main__":
    count_implant(sys.argv)
