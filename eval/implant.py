import pandas as pd

df_implant = pd.read_csv("/mnt/hdd/PANO.arlen/google_sheet/PANO_implant.csv")
df_finding = pd.read_csv("/mnt/hdd/PANO.arlen/google_sheet/PANO_finding.csv")
df_result = pd.read_csv("/mnt/hdd/PANO.arlen/results/2023-07-23-014113/count_implant.csv")

df_finding.finding = df_finding.finding == "IMPLANT"
df_golden = df_finding.groupby('file_name')['finding'].sum().reset_index()

# remove exclude
names_to_remove = df_implant.loc[df_implant["exclude"] == True, "file_name"]
df_golden = df_golden[~df_golden["file_name"].isin(names_to_remove)]
df_result = df_result[~df_result["file_name"].isin(names_to_remove)]

merged_df = df_golden.merge(df_result, on='file_name', how='left')

acc = sum(merged_df["finding"] == merged_df["num_implant"])/len(merged_df)
print(acc)

wrong = merged_df.loc[merged_df["finding"] != merged_df["num_implant"], ["file_name", "finding", "num_implant"]]
df_result: pd.DataFrame = pd.DataFrame(wrong)
df_result.to_csv("/mnt/hdd/PANO.arlen/google_sheet/wrong.csv", index=False)