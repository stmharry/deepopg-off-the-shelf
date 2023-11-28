import pandas as pd

# Load the datasets
pano_finding_1_df = pd.read_csv(
    "/mnt/hdd/PANO.arlen/data/raw_data/Pano Finding - 1 (Responses) - Form Responses 1.csv"
)
pano_finding_2_df = pd.read_csv(
    "/mnt/hdd/PANO.arlen/data/raw_data/Pano Finding - 2 (Responses) - Form responses 1.csv"
)
pano_finding_3_df = pd.read_csv(
    "/mnt/hdd/PANO.arlen/data/raw_data/Pano Finding - 3 (Responses) - Form responses 1.csv"
)
pano_finding_4_df = pd.read_csv(
    "/mnt/hdd/PANO.arlen/data/raw_data/Pano Finding - 4 (Responses) - Form responses 1.csv"
)

pano_image_1_df = pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/1.csv", header=None)
pano_image_2_df = pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/2.csv", header=None)
pano_image_3_df = pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/3.csv", header=None)
pano_image_4_df = pd.read_csv("/mnt/hdd/PANO.arlen/data/raw_data/4.csv", header=None)

pano_finding = [
    [pano_finding_1_df, pano_image_1_df],
    [pano_finding_2_df, pano_image_2_df],
    [pano_finding_3_df, pano_image_3_df],
    [pano_finding_4_df, pano_image_4_df],
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
dict_finding_dentist = dict(zip(list_finding_dentist, list_finding_PANO))


# Helper function to prepare the dataframe for each doctor
def prepare_dataframe(df, image_df, doctor):
    # Filter the dataframe for the specific doctor
    doctor_df = df[df["Your Number"] == doctor]

    # Melt the dataframe to make it long-form
    melted_df = doctor_df.melt(
        id_vars=["Your Number"], var_name="fdi", value_name="Finding"
    )

    # Remove rows where Finding is NaN
    melted_df = melted_df.dropna(subset=["Finding"])

    # Split findings into separate rows if there are multiple findings for one tooth
    findings_separated = melted_df["Finding"].str.split(", ", expand=True).stack()

    # Index levels to align separated findings with the original dataframe
    idx = findings_separated.index.droplevel(-1)

    # Create a new dataframe with separated findings
    separated_df = melted_df.loc[idx, ["Your Number", "fdi"]].copy()
    separated_df["finding"] = findings_separated.values

    # Map the 'finding' to the new names using the provided dictionary
    separated_df["finding"] = separated_df["finding"].map(dict_finding_dentist)

    # Extract the image number from the tooth code column
    # This time, we ensure the extracted number is a string to avoid NaN issues.
    separated_df["Image Number"] = separated_df["fdi"].str.extract(r"^(\d+)-")[0]
    separated_df["fdi"] = separated_df["fdi"].str.extract(r"\[(\d+)\]")[0]

    # Convert image numbers to integers where possible
    separated_df["Image Number"] = (
        pd.to_numeric(separated_df["Image Number"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Map image numbers to file names
    separated_df = separated_df.merge(
        image_df, how="left", left_on="Image Number", right_on=image_df.columns[0]
    )
    separated_df[image_df.columns[1]] = separated_df[image_df.columns[1]].str.replace(
        ".png", "", regex=False
    )

    # Rename columns to match required output
    separated_df.rename(columns={image_df.columns[1]: "file_name"}, inplace=True)

    # Drop unnecessary columns
    separated_df.drop(columns=["Your Number", "Image Number"], inplace=True)

    # Reorder columns
    separated_df = separated_df[["file_name", "fdi", "finding"]]

    return separated_df.reset_index(drop=True)


# Function to clean up the dataframe by removing metadata rows and ensuring 'Image Name' is correct
def clean_dataframe(df):
    # Remove rows where 'Tooth Code' is not in the expected format (like 'Timestamp')
    df_cleaned = df.dropna(subset=["fdi"])

    # Drop any rows that still have NaN values in 'Image Name' after the merge
    df_cleaned = df_cleaned.dropna(subset=["file_name"])

    return df_cleaned


def combine_dataframes_by_doctor(pano_finding_list):
    # Initialize dictionaries to hold combined dataframes for each doctor
    combined_dfs = {
        "A": pd.DataFrame(),
        "C": pd.DataFrame(),
        "D": pd.DataFrame(),
        "E": pd.DataFrame(),
    }

    # Loop through each set of findings and images
    for pano_finding_df, image_names_df in pano_finding_list:
        # Get unique doctors from the current finding dataframe
        doctors = pano_finding_df["Your Number"].unique()
        # Prepare and clean dataframes for each doctor and combine with the previous ones
        for doctor in doctors:
            cleaned_df = clean_dataframe(
                prepare_dataframe(pano_finding_df, image_names_df, doctor)
            )
            combined_dfs[doctor] = pd.concat(
                [combined_dfs[doctor], cleaned_df], ignore_index=True
            )

    return combined_dfs


if __name__ == "__main__":
    # Combine the dataframes for each doctor across all CSV files
    combined_doctor_dfs = combine_dataframes_by_doctor(pano_finding)

    # You can do similar for doctors 'A', 'C', 'D', and 'E'
    combined_doctor_dfs["A"].to_csv(
        "/mnt/hdd/PANO.arlen/data/meta_data/doctor_a_combined.csv", index=False
    )
    combined_doctor_dfs["C"].to_csv(
        "/mnt/hdd/PANO.arlen/data/meta_data/doctor_c_combined.csv", index=False
    )
    combined_doctor_dfs["D"].to_csv(
        "/mnt/hdd/PANO.arlen/data/meta_data/doctor_d_combined.csv", index=False
    )
    combined_doctor_dfs["E"].to_csv(
        "/mnt/hdd/PANO.arlen/data/meta_data/doctor_e_combined.csv", index=False
    )
