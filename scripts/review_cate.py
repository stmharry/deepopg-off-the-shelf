import pandas as pd

# with open('/mnt/md0/data/PANO/data/raw/NTUH/ntuh-opg-12.json', 'r') as json_file:
#     data = json.load(json_file)

# images = data["images"]
# file_name = [item["file_name"].split(".")[0].split("_")[0] for item in images]
# print(pd.value_counts(file_name))

pano_golden = pd.read_csv(
    "/home/arlen/deepopg-eval/(Frozen) Promaton Metadata + Summary Golden Label - Per-finding.csv"
)
golden_split = pd.read_csv(
    "/mnt/md0/data/PANO/data/splits/instance-detection-v1/eval.txt", header=None
)
# remove the "PROMATON" front the golden_split
golden_split = golden_split[0].apply(lambda x: x.split("/")[-1].split("_")[0])

# missing_list
missing_list = (
    pano_golden[pano_golden["finding"] == "MISSING"]["file_name"].unique().tolist()
)

# no missing list = golden_split not in missing_list
cate1_list = golden_split[~golden_split.isin(missing_list)].tolist()
cate1 = len(cate1_list)

# finding = "IMPLANT"
cate4_list = (
    pano_golden[pano_golden["finding"] == "IMPLANT"]["file_name"].unique().tolist()
)
cate4 = len(cate4_list)

# new missing_list = missing_list - cate4_list
missing_list = list(set(missing_list) - set(cate4_list))

# have crown_bridge
crown_bridge_list = (
    pano_golden[pano_golden["finding"] == "CROWN_BRIDGE"]["file_name"].unique().tolist()
)
cate3_list = list(set(crown_bridge_list) & set(missing_list))
cate3 = len(cate3_list)

# new missing_list = missing_list - cate3_list
missing_list = list(set(missing_list) - set(cate3_list))

# pano_golden finding all missing
pano_golden_missing = pano_golden[pano_golden["file_name"].isin(missing_list)]
# each file_name's finding only "MISSING"
pano_golden_missing = pano_golden_missing.groupby("file_name").filter(
    lambda x: x["finding"].nunique() == 1
)
cate5_list = pano_golden_missing["file_name"].unique().tolist()
cate5 = len(cate5_list)

# cate2 = missing_list - cate5_list
cate2_list = list(set(missing_list) - set(cate5_list))
cate2 = len(cate2_list)

# print cate1, cate2, cate3, cate4, cate5
print(cate1, cate2, cate3, cate4, cate5)
