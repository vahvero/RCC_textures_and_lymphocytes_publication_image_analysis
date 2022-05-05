# pylint: disable=all
__author__ = "MSc. Otso Brummer, <https://github.com/vahvero>"
__date__ = "2022-05-5"

# %% Imports
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from utils import extract_margin

tissue_mapping = {
    "empty": 0,
    "blood": 1,
    "cancer": 2,
    "normal": 3,
    "stroma": 4,
    "other": 5,
}


def vote_filter(
    tensor,
    window=(3, 3),
):
    """Window voting filter with bias

    :param tensor: Classification image
    :type tensor: np.array
    :param window: Size of the filter, defaults to (3, 3)
    :type window: tuple[int, int], optional
    :return: Filtered classification image
    :rtype: np.array
    """
    out = np.copy(tensor)
    center_index = window[0] // 2, window[1] // 2

    # 1 to N - 1
    for i in range(center_index[0], tensor.shape[0] - center_index[0]):
        for j in range(center_index[1], tensor.shape[1] - center_index[1]):
            # Access is [-1 to +1]
            window = tensor[
                i - center_index[0] : i + center_index[0] + 1,
                j - center_index[1] : j + center_index[1] + 1,
            ]

            uniq, counts = np.unique(window, return_counts=True)
            count_max = np.max(counts)
            maxs = uniq[counts == count_max]
            maxs = maxs[maxs.nonzero()]
            if maxs.size == 0:
                continue

            # Stroma and cancer bias
            if tissue_mapping["cancer"] in maxs:
                new_max_idx = tissue_mapping["cancer"]
            elif tissue_mapping["stroma"] in maxs:
                new_max_idx = tissue_mapping["stroma"]
            else:
                new_max_idx = np.random.choice(maxs[maxs.nonzero()], 1)

            # Do not care for the margins
            out[i, j] = new_max_idx

    return out


# %% Constants

result_folder = "results"
ignore_list = {
    # Files with low magnification
    "TCGA-CZ-4861",
    "TCGA-CZ-4858",
    "TCGA-B8-5164",
    "TCGA-B8-5163",
    "TCGA-B8-5165",
    "TCGA-CZ-4854",
    "TCGA-CZ-4864",
    "TCGA-B8-5159",
    "TCGA-CZ-4856",
    "TCGA-B8-5158",
    "TCGA-BP-4770",
    "TCGA-CZ-4857",
    "TCGA-CZ-4865",
    "TCGA-CZ-4859",
    "TCGA-CZ-4866",
    "TCGA-CZ-4863",
    "TCGA-CZ-4862",
    "TCGA-CZ-4853",
}

ym2_to_mm2 = 1e6 / 15_500
guesses_filename = "tissue_classification_predictions.pth"
tissue_filename = "tissue_classification.pth"
lymphocyte_filename = "lymphocytes.pth"
general_cells_filename = "generic_cells.pth"
binary_lymphs_filename = "binary_lymphocytes.pth"

data_pickle = "rcc_analysis.pickle"
mmc2_file = "mmc2.xlsx"
raw_data_file = "raw_data.xlsx"

# %% Load data
try:
    with open(data_pickle, "rb") as fobj:
        data = pickle.load(fobj)
except FileNotFoundError:
    data_dict = defaultdict(list)
    results_folders = os.listdir(result_folder)
    for idx, folder in enumerate(results_folders):
        if folder in ignore_list:
            continue
        total_folder = f"{result_folder}/{folder}"

        tissue_guesses = torch.load(total_folder + "/" + guesses_filename)
        tissue = torch.load(total_folder + "/" + tissue_filename)
        binary_lymphs = torch.load(total_folder + "/" + binary_lymphs_filename)

        # Take top 2 guesses
        values, indices = tissue_guesses.topk(2)

        # If differences between top 2 guesses is smaller than 2
        cancer_class = (
            torch.abs(torch.diff(values, dim=2)).squeeze()
            < 2
            # And one of the types is cancer
        ) & torch.any(tissue_mapping["cancer"] == indices, dim=2)
        # Set these indices to cancer
        tissue[cancer_class] = tissue_mapping["cancer"]
        tissue = vote_filter(
            tissue.numpy(),
            window=(
                3,
                3,
            ),
        )

        # tissue = signal.medfilt2d(tissue, kernel_size=(3,3,))
        tissue = torch.from_numpy(tissue)

        data_dict["tcga_id"].append(folder)
        folder = total_folder

        margin_index = extract_margin(
            tissue,
            tissue_mapping,
            (
                5,
                5,
            ),
        )

        for tissue_name, t_type in tissue_mapping.items():

            # Tissue counts
            data_dict[f"texture_{tissue_name}"].append(
                int(torch.sum(tissue == t_type)),
            )

            data_dict[f"bin_lymphocytes_{tissue_name}"].append(
                float(torch.sum(binary_lymphs[tissue == t_type][:, 0]))
            )

            data_dict[f"bin_generic_{tissue_name}"].append(
                float(torch.sum(binary_lymphs[tissue == t_type][:, 1]))
            )

            # Margin counts
            data_dict[f"margin_texture_{tissue_name}"].append(
                int(torch.sum((tissue[margin_index] == t_type)))
            )

            data_dict[f"margin_bin_lymphocytes_{tissue_name}"].append(
                float(torch.sum(binary_lymphs[(tissue == t_type) & margin_index][:, 0]))
            )

            data_dict[f"margin_bin_generic_{tissue_name}"].append(
                float(torch.sum(binary_lymphs[(tissue == t_type) & margin_index][:, 1]))
            )

            # Non Margin counts
            data_dict[f"non_margin_texture_{tissue_name}"].append(
                int(torch.sum((tissue[~margin_index] == t_type)))
            )

            data_dict[f"non_margin_bin_lymphocytes_{tissue_name}"].append(
                float(
                    torch.sum(binary_lymphs[(tissue == t_type) & ~margin_index][:, 0])
                )
            )
            data_dict[f"non_margin_bin_generic_{tissue_name}"].append(
                float(
                    torch.sum(binary_lymphs[(tissue == t_type) & ~margin_index][:, 1])
                )
            )
        print(f"{idx + 1} / {len(results_folders)} {folder} done")
    # Gather to dataframe
    data = pd.DataFrame.from_dict(data_dict)
    data = data.set_index("tcga_id")

    # Calculate percentages
    data["texture_total"] = data.filter(like="texture_").sum(axis=1)
    data["margin_texture_total"] = data.filter(like="margin_texture_").sum(axis=1)
    data["non_margin_texture_total"] = data.filter(like="non_margin_texture_").sum(
        axis=1
    )

    for tissue_name, t_type in tissue_mapping.items():
        data[f"texture_{tissue_name}_%"] = (
            data[f"texture_{tissue_name}"] / data["texture_total"]
        ).fillna(0)

        data[f"margin_texture_{tissue_name}_%"] = (
            data[f"margin_texture_{tissue_name}"] / data["margin_texture_total"]
        ).fillna(0)

        data[f"non_margin_texture_{tissue_name}_%"] = (
            data[f"non_margin_texture_{tissue_name}"] / data["non_margin_texture_total"]
        ).fillna(0)

        data[f"inf_bin_lymphocytes_{tissue_name}"] = (
            data[f"bin_lymphocytes_{tissue_name}"] / data[f"texture_{tissue_name}"]
        ).fillna(0)

        data[f"inf_margin_bin_lymphocytes_{tissue_name}"] = (
            data[f"margin_bin_lymphocytes_{tissue_name}"]
            / data[f"margin_texture_{tissue_name}"]
        ).fillna(0)

        data[f"inf_non_margin_bin_lymphocytes_{tissue_name}"] = (
            data[f"non_margin_bin_lymphocytes_{tissue_name}"]
            / data[f"non_margin_texture_{tissue_name}"]
        ).fillna(0)

    # Irrelevant
    data = data.drop(
        columns=[
            "inf_margin_bin_lymphocytes_cancer",
        ]
    )
    data.to_excel(raw_data_file)
    # Read external data
    mmc2 = pd.read_excel(mmc2_file)
    mmc2 = mmc2.loc[~mmc2["bcr_patient_barcode"].isna()]
    mmc2 = mmc2.set_index("bcr_patient_barcode")
    mmc2 = mmc2[mmc2["PanKidney Pathology"] == "ccRCC"]
    mmc2["Survival"] = pd.to_numeric(mmc2["Survival"])
    mmc2 = mmc2.drop("Unnamed: 0", axis=1)

    current = data.shape[0]
    data = pd.concat(
        [
            mmc2,
            data,
        ],
        join="inner",
        axis=1,
    )
    data["tissue_source_site"] = [x[5:7] for x in data.index]
    print(f"MMC2 dropped N {current} to {data.shape[0]}")

    # Pickle resulting dataframe to speed up subscuent runs
    with open(data_pickle, "wb") as fobj:
        pickle.dump(data, fobj)

# %%
