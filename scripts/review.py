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
df_golden = pd.read_csv("/mnt/hdd/PANO.arlen/golden/ntuh_golden.csv")
df_doctor = [
    pd.read_csv("/mnt/hdd/PANO.arlen/data/meta_data/doctor_a_combined.csv"),
    pd.read_csv("/mnt/hdd/PANO.arlen/data/meta_data/doctor_c_combined.csv"),
    pd.read_csv("/mnt/hdd/PANO.arlen/data/meta_data/doctor_d_combined.csv"),
    pd.read_csv("/mnt/hdd/PANO.arlen/data/meta_data/doctor_e_combined.csv"),
]

image_name = [
    pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/1.csv", header=None),
    pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/2.csv", header=None),
    pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/3.csv", header=None),
    pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/4.csv", header=None),
]


def doctor_image_list(
    image_name: list,
) -> list:
    image_list = pd.DataFrame()
    for image in image_name:
        image[1] = image[1].str.replace(".png", "", regex=False)
        image_list = pd.concat([image_list, image[1]], ignore_index=True)

    return image_list[0].tolist()


def ROC_curve(
    df_golden: pd.DataFrame,
    df_result: pd.DataFrame,
    list_dentist: list,
) -> list:
    list_dentist = list_dentist
    fpr_finding: list = []
    tpr_finding: list = []
    auc_finding: list = []
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
                df_full_tooth["file_name"] = i

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
                df_full_tooth["file_name"] = i

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

        df_auc.to_csv("tmp/{}_roc.csv".format(finding_name))
        golden = df_auc["golden"]
        score = df_auc["score"]
        confusion_arr = confusion_matrix(golden, score != 0)
        fpr, tpr, thresholds = roc_curve(golden, score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        fpr_finding.append(fpr)
        tpr_finding.append(tpr)
        auc_finding.append(roc_auc)

    return fpr_finding, tpr_finding, auc_finding


def confus_dentist(
    df_golden: pd.DataFrame, df_dentist: pd.DataFrame, dentist_image_list: list
) -> list:
    TPR: list = []
    FPR: list = []
    for finding_name in list_finding_PANO:
        df_golden_finding = df_golden[df_golden["finding"] == finding_name]
        df_dentist_finding = df_dentist[df_dentist["finding"] == finding_name]

        df_review = pd.DataFrame()
        for i in dentist_image_list:
            file_golden = df_golden_finding[df_golden_finding["file_name"] == i]
            file_dentist = df_dentist_finding[df_dentist_finding["file_name"] == i]

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
            df_full_tooth["dentist"] = (
                df_full_tooth["fdi"].isin(file_dentist["fdi"]).astype(int)
            )
            df_review = pd.concat([df_review, df_full_tooth], ignore_index=True)

        tn, fp, fn, tp = confusion_matrix(
            df_review["golden"], df_review["dentist"]
        ).ravel()
        TPR.append(tp / (tp + fn))
        FPR.append(fp / (fp + tn))

    return TPR, FPR


if __name__ == "__main__":
    golden_dataframe = df_golden
    TPR_alldentist: list = []
    FPR_alldentist: list = []
    dentist_image_list = doctor_image_list(image_name)
    for id, dentist in enumerate(df_doctor):
        TPR, FPR = confus_dentist(golden_dataframe, dentist, dentist_image_list)
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
