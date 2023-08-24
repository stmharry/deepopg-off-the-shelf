import json
import pandas as pd
import re

row_results = []
ntuh_sheet = pd.read_csv("/mnt/hdd/PANO.arlen/google_sheet/NTUH_sheet.csv")

with open('/mnt/hdd/PANO/data/coco/instance-detection-v1-ntuh.json') as json_file:
    data = json.load(json_file)

images = data["images"]
finding = data["categories"]

lookup_file = {item["id"]: item["file_name"].split("/")[1].split(".")[0] for item in images}
lookup_finding = {item["name"]: item["id"] for item in finding}

def translate_file(ntuh_sheet_no):
    return lookup_file.get(ntuh_sheet_no, None)

def us_fdi(x: int):
    grid = (x-1) // 8 + 1
    index = (x-1) % 8 + 1 if grid % 2 == 0 else 9 - ((x-1) % 8 + 1)
    fdi = grid * 10 + index
    return fdi

list_finding = ["Missing", "Implant", "Remnant Root", "Caries", "Root Filled", "Crown Bridge", "Apical Lesion", "Restoration"]
list_finding_PANO = ["MISSING", "IMPLANT", "ROOT_REMNANTS", "CARIES", "ENDO", "CROWN_BRIDGE", "PERIAPICAL_RADIOLUCENT", "FILLING"]

dict_finding = dict(zip(list_finding, list_finding_PANO))

for i, id in enumerate(ntuh_sheet["No."]):
    file_name = translate_file(id)
    if file_name is None:
        print("No tooth: {}".format(id))
        continue

    for f in list_finding:
        cell_value = ntuh_sheet[f][i]
        if isinstance(cell_value, str) and not pd.isna(cell_value):
            matches = re.findall(r'\[(\d+)\]', cell_value)
            tooth_us = [int(num) for num in matches]
            for j in tooth_us:
                fdi = us_fdi(j)
                row_results.append(
                    {
                        "file_name": file_name,
                        "fdi": fdi,
                        "finding": dict_finding[f]
                    }
                )

# create csv
df_result: pd.DataFrame = pd.DataFrame(row_results)
df_result.to_csv("ntuh_golden.csv", index=False)
