import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

list_finding = [
    "Missing",
    "Implant",
    "Remnant Root",
    "Caries",
    "Root Filled",
    "Crown Bridge",
    "Apical Lesion",
    "Restoration",
]
list_finding_dentist = [
    "Missing",
    "Implant",
    "Remnant Root",
    "Caries",
    "Root Filled",
    "Crown Bridge",
    "Apical Lesion",
    "Restorations",
]
list_finding_PANO = [
    "MISSING",
    "IMPLANT",
    "ROOT_REMNANTS",
    "CARIES",
    "ENDO",
    "CROWN_BRIDGE",
    "PERIAPICAL_RADIOLUCENT",
    "FILLING",
]
dict_finding = dict(zip(list_finding, list_finding_PANO))
dict_finding_dentist = dict(zip(list_finding_dentist, list_finding_PANO))
df_result = pd.read_csv("/mnt/hdd/PANO.arlen/results/2023-10-16-083146/result.csv")


def us_fdi(x: int):
    grid = (x - 1) // 8 + 1
    index = (x - 1) % 8 + 1 if grid % 2 == 0 else 9 - ((x - 1) % 8 + 1)
    fdi = grid * 10 + index
    return fdi


def Golden(dataset_name: str = "ntuh") -> pd.DataFrame:
    if dataset_name == "ntuh":
        sheet = pd.read_csv(
            "/home/arlen/deepopg-eval/review/(WIP) NTUH Summary Golden Label - Per-study.csv"
        )
        with open(
            "/mnt/md0/data/PANO/data/raw/NTUH/ntuh-opg-12.json", "r"
        ) as json_file:
            data = json.load(json_file)
    images = data["images"]
    lookup_file = {item["id"]: item["file_name"].split(".")[0] for item in images}

    row_results: list = []
    for i, id in enumerate(sheet["No."]):
        file_name = lookup_file.get(id, None)
        if file_name is None:
            continue

        for f in list_finding:
            cell_value = sheet[f][i]
            if isinstance(cell_value, str) and not pd.isna(cell_value):
                matches = re.findall(r"\[(\d+)\]", cell_value)
                tooth_us = [int(num) for num in matches]
                for j in tooth_us:
                    fdi = us_fdi(j)
                    if f == "Implant":
                        row_results.append(
                            {"file_name": file_name, "fdi": fdi, "finding": "MISSING"}
                        )
                    row_results.append(
                        {"file_name": file_name, "fdi": fdi, "finding": dict_finding[f]}
                    )

    # create csv
    df_golden: pd.DataFrame = pd.DataFrame(row_results)

    return df_golden


def dentist_review(
    dentist_name: str = "A",
) -> None:
    pano_finding_1_df = pd.read_csv(
        "/home/arlen/deepopg-eval/review/Pano Finding - 1 (Responses) - Form Responses 1.csv"
    )
    pano_finding_2_df = pd.read_csv(
        "/home/arlen/deepopg-eval/review/Pano Finding - 2 (Responses) - Form responses 1.csv"
    )
    pano_finding_3_df = pd.read_csv(
        "/home/arlen/deepopg-eval/review/Pano Finding - 3 (Responses) - Form responses 1.csv"
    )
    pano_finding_4_df = pd.read_csv(
        "/home/arlen/deepopg-eval/review/Pano Finding - 4 (Responses) - Form responses 1.csv"
    )

    pano_image_1_df = pd.read_csv("/home/arlen/deepopg-eval/review/1.csv", header=None)
    pano_image_2_df = pd.read_csv("/home/arlen/deepopg-eval/review/2.csv", header=None)
    pano_image_3_df = pd.read_csv("/home/arlen/deepopg-eval/review/3.csv", header=None)
    pano_image_4_df = pd.read_csv("/home/arlen/deepopg-eval/review/4.csv", header=None)

    pano_finding = [
        [pano_finding_1_df, pano_image_1_df],
        [pano_finding_2_df, pano_image_2_df],
        [pano_finding_3_df, pano_image_3_df],
        [pano_finding_4_df, pano_image_4_df],
    ]
    dentist_list = []
    # total 120 images
    image_list = pd.concat(
        [image[1] for df, image in pano_finding], ignore_index=True
    ).values.tolist()
    dentist_image_list = [filename.replace(".png", "") for filename in image_list]
    # only 117 images was reviewed by dentists
    for df, image in pano_finding:
        for tooth_name in range(1, 32):
            mask = df.columns.str.startswith("{}-".format(tooth_name))
            your_number = df["Your Number"]
            result = pd.concat([your_number, df.loc[:, mask]], axis=1)
            filtered_result = result[result["Your Number"] == dentist_name]
            for i in filtered_result.columns[1:]:
                if filtered_result[i].isin(list_finding_dentist).any():
                    dentist_list.append(
                        {
                            "file_name": image[1][tooth_name - 1].split(".")[0],
                            "fdi": int(i.split("[")[1].split("]")[0]),
                            "finding": dict_finding_dentist[
                                filtered_result[i].values[0]
                            ],
                        }
                    )
    df_result: pd.DataFrame = pd.DataFrame(dentist_list)
    return df_result, dentist_image_list


def ROC_curve(
    df_golden: pd.DataFrame,
    df_result: pd.DataFrame,
    list_dentist: list,
) -> list:
    list_dentist = list_dentist
    fpr_finding: list = []
    tpr_finding: list = []
    auc_finding: list = []
    list_finding_PANO = ["ENDO"]
    for finding_name in list_finding_PANO:
        df_auc = pd.DataFrame()
        # missing tooth
        if finding_name == "MISSING" or finding_name == "IMPLANT":
            df_golden_finding = df_golden[df_golden["finding"] == finding_name]
            df_result_finding = df_result[df_result["finding"] == finding_name]
            for i in list_dentist:
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
                file_result.set_index("fdi", inplace=True)
                df_full_tooth.set_index("fdi", inplace=True)
                df_full_tooth["score"] = file_result["score"]
                df_full_tooth.reset_index(inplace=True)
                df_full_tooth["score"].fillna(0, inplace=True)

                df_auc = pd.concat([df_auc, df_full_tooth], ignore_index=True)
        # no missing tooth
        else:
            df_golden_missing = df_golden[df_golden["finding"] == "MISSING"]
            df_golden_finding = df_golden[df_golden["finding"] == finding_name]
            df_result_finding = df_result[df_result["finding"] == finding_name]

            for i in list_dentist:
                file_missing = df_golden_missing[df_golden_missing["file_name"] == i]
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

                file_result.set_index("fdi", inplace=True)
                df_full_tooth.set_index("fdi", inplace=True)
                df_full_tooth["score"] = file_result["score"]
                df_full_tooth.reset_index(inplace=True)
                df_full_tooth["score"].fillna(0, inplace=True)

                df_full_tooth["missing"] = (
                    df_full_tooth["fdi"].isin(file_missing["fdi"]).astype(int)
                )
                df_full_tooth = df_full_tooth[df_full_tooth["missing"] == 0].drop(
                    columns=["missing"]
                )

                # save the dataframe with golden and score without missing
                df_auc = pd.concat([df_auc, df_full_tooth], ignore_index=True)

        # df_auc.to_csv("tmp/{}.csv".format(finding_name))
        # breakpoint()
        golden = df_auc["golden"]
        score = df_auc["score"]
        # confusion_arr = confusion_matrix(golden, score != 0)
        # print(confusion_arr)
        fpr, tpr, thresholds = roc_curve(golden, score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        fpr_finding.append(fpr)
        tpr_finding.append(tpr)
        auc_finding.append(roc_auc)

    return fpr_finding, tpr_finding, auc_finding


def confus_dentist(df_golden: pd.DataFrame, df_dentist: pd.DataFrame) -> list:
    TPR: list = []
    FPR: list = []
    file_list = pd.unique(df_dentist["file_name"]).tolist()
    for finding_name in list_finding_PANO:
        df_golden_finding = df_golden[df_golden["finding"] == finding_name]
        df_dentist_finding = df_dentist[df_dentist["finding"] == finding_name]

        df_review = pd.DataFrame()
        for i in file_list:
            file_golden = df_golden_finding[df_golden_finding["file_name"] == i]
            file_dentist = df_dentist_finding[df_dentist_finding["file_name"] == i]

            df_full_tooth: pd.DataFrame = pd.DataFrame(
                [
                    {"fdi": int(f"{quadrant}{tooth}")}
                    for quadrant in range(1, 5)
                    for tooth in range(1, 9)
                ]
            )

            df_full_tooth["file_name"] = i
            df_full_tooth["golden"] = (
                df_full_tooth["fdi"].isin(file_golden["fdi"]).astype(int)
            )
            df_full_tooth["dentist"] = (
                df_full_tooth["fdi"].isin(file_dentist["fdi"]).astype(int)
            )
            df_review = pd.concat([df_review, df_full_tooth], ignore_index=True)

        confusion_arr = confusion_matrix(df_review["golden"], df_review["dentist"])
        TPR.append(confusion_arr[0, 0] / (confusion_arr[0, 0] + confusion_arr[1, 0]))
        FPR.append(confusion_arr[0, 1] / (confusion_arr[0, 1] + confusion_arr[1, 1]))

    return TPR, FPR


if __name__ == "__main__":
    golden_dataframe = Golden(dataset_name="ntuh")
    dentist_name = ["A", "C", "D", "E"]
    TPR_alldentist: list = []
    FPR_alldentist: list = []
    for dentist in dentist_name:
        dentist_dataframe, dentist_image_list = dentist_review(dentist_name=dentist)
        TPR, FPR = confus_dentist(golden_dataframe, dentist_dataframe)
        TPR_alldentist.append(TPR)
        FPR_alldentist.append(FPR)

    fpr, tpr, roc_auc = ROC_curve(golden_dataframe, df_result, dentist_image_list)

    for id, finding_name in enumerate(list_finding_PANO):
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr[id],
            tpr[id],
            color="darkorange",
            lw=2,
            label="ROC curve (AUC = {:.2f})".format(roc_auc[id]),
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.scatter(FPR_alldentist[0][id], TPR_alldentist[0][id], c="r")
        plt.scatter(FPR_alldentist[1][id], TPR_alldentist[1][id], c="b")
        plt.scatter(FPR_alldentist[2][id], TPR_alldentist[2][id], c="g")
        plt.scatter(FPR_alldentist[3][id], TPR_alldentist[3][id], c="m")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("{} Receiver Operating Characteristic (ROC)".format(finding_name))
        plt.legend(loc="lower right")
        plt.savefig("image/{}.png".format(finding_name))
