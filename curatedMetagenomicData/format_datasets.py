import copy
import os
from pathlib import Path

import pandas as pd

metadata_folder = "./metadata"

metadata = dict()

for metadata_filepath in Path(metadata_folder).glob("*.tsv"):
    with open(metadata_filepath) as meta:
        header = meta.readline().strip().split("\t")
        for line in meta:
            line = line.strip()
            if line:
                line_split = line.split("\t")
                dataset = line_split[header.index("study_name")]
                if dataset not in metadata:
                    metadata[dataset] = dict()
                sample_id = line_split[header.index("sample_id")]
                metadata[dataset][sample_id] = dict()

                for h in header:
                    metadata[dataset][sample_id][h] = line_split[header.index(h)]

selection = dict()

for dataset in metadata:
    selection[dataset] = dict()

    values = dict()

    # Remove samples under antibiotics
    for sample_id in copy.deepcopy(metadata[dataset]):
        if "antibiotics_current_use" in metadata[dataset][sample_id] and metadata[dataset][sample_id]["antibiotics_current_use"] == "yes":
            del metadata[dataset][sample_id]

    for sample_id in metadata[dataset]:
        metas = list(metadata[dataset][sample_id].keys())

        if "disease" in metas and "study_condition" in metas:
            # Consider study_condition only
            metas.remove("disease")

        if "DNA_extraction_kit" in metas:
            metas.remove("DNA_extraction_kit")

        if "curator" in metas:
            metas.remove("curator")

        for meta in metas:
            if meta not in values:
                values[meta] = list()

            values[meta].append(metadata[dataset][sample_id][meta])

    for meta in values:
        unique = set(values[meta])
        unique.discard("NA")

        is_not_digit = all(not value.isdigit() for value in unique)

        # Consider categorical metadata only with 2 values
        if is_not_digit and len(unique) == 2:
            # Check whether the number of samples in the two classes is "balanced"
            # The class with the minimum number of samples shouldn't contain less than 30% of sample
            # The total number of samples shouldn't be lower than 25
            unique_list = list(unique)
            samples = [values[meta].count(unique_list[0]), values[meta].count(unique_list[1])]

            total_samples = sum(samples)
            min_samples = min(samples)
            min_samples_perc = round(float(min_samples) * 100.0 / float(total_samples))

            if total_samples >= 25 and min_samples_perc >= 30:
                selection[dataset][meta] = dict()

                for sample_id in metadata[dataset]:
                    if meta in metadata[dataset][sample_id] and metadata[dataset][sample_id][meta] != "NA":
                        selection[dataset][meta][sample_id] = metadata[dataset][sample_id][meta]

    if not selection[dataset]:
        del selection[dataset]

relative_abundances_folder = "./relative_abundance"

output_datasets_folder = "./relative_abundance_classification"
os.makedirs(output_datasets_folder, exist_ok=True)

for relative_abundance_filepath in Path(relative_abundances_folder).glob("*.tsv"):
    dataset = os.path.splitext(os.path.basename(str(relative_abundance_filepath)))[0].split(".")[1]
    if dataset in selection:
        print("Processing {}".format(dataset))

        for meta in selection[dataset]:
            df = pd.read_csv(str(relative_abundance_filepath), sep="\t")

            df['sample_id'] = df.index
            cols = df.columns.tolist()

            # Do not use the whole taxonomic label
            # Use the species name only
            cols = [c.split("|")[-1][3:] if c.split("|")[-1].startswith("s__") else c for c in cols]
            df.columns = cols

            cols = cols[-1:] + cols[:-1]
            df = df.loc[:, cols]

            df['label'] = [selection[dataset][meta][sample_id] if sample_id in selection[dataset][meta] else "NA" for sample_id in df.index.tolist()]

            # Drop samples without a class label
            df = df.drop(df[df.label == "NA"].index)

            # Cound the number of samples in the two classes again
            classes = list(set(df.label.tolist()))

            assert len(classes) == 2

            samples = [len(df[df.label == classes[0]].index), len(df[df.label == classes[1]].index)]

            total_samples = sum(samples)
            min_samples = min(samples)
            min_samples_perc = round(float(min_samples) * 100.0 / float(total_samples))

            if total_samples >= 25 and min_samples_perc >= 30:
                print(
                    "\t{} ({}:{}; {}:{})".format(
                        meta,
                        classes[0],
                        len(df[df.label == classes[0]].index),
                        classes[1],
                        len(df[df.label == classes[1]].index)
                    )
                )

                df.to_csv(
                    os.path.join(output_datasets_folder, "{}.{}.tsv".format(os.path.splitext(os.path.basename(str(relative_abundance_filepath)))[0], meta)),
                    sep="\t", index = False
                )
