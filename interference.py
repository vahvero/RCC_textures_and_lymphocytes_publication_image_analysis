"""Run model evaluation over all files in ``data_folder``.
See TODOs to change behaviour.
"""

__author__ = "MSc. Otso Brummer, <https://github.com/vahvero>"
__date__ = "2022-05-5"

# %% Imports

# pylint: disable=all
import glob
import logging
import os

import torch
from utils import SlideDataset
from torch import nn
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm

# %%

force_classification = False
force_binary_recognition = False
verbose = False

models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}

rcc_folder = "/mnt/data/RCC"
"""Folder where tcga SVS files reside"""

logging.basicConfig(
    filename="interference.log",
    filemode="w",
    format="[%(asctime)s]: %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
)

logger = logging.getLogger("interference_logger")
logger.setLevel(logging.DEBUG)
logger.info(
    f"Executing with"
    + f" force_classification={force_classification},"
    + f"force_binary_recognition={force_binary_recognition}"
)
logging.info("Loading models")
# Find all files
device = torch.device("cuda:0")

classification_class_n = 6
classification_model = models["resnet18"](
    pretrained=False,
)
features = classification_model.fc.in_features
classification_model.fc = nn.Sequential(
    nn.Linear(features, features),
    nn.ReLU(),
    nn.Linear(features, classification_class_n),
)
# TODO: Change to texture classification model path
classification_model.load_state_dict(
    torch.load(
        "resnet18_tissue_classfication.pth",
        map_location="cpu",
    )
)
classification_model = classification_model.to(device)
classification_model.eval()

binary_recognition_model = models["resnet18"](
    pretrained=False,
)
features = binary_recognition_model.fc.in_features
binary_recognition_model.fc = nn.Sequential(
    nn.Linear(features, features),
    nn.ReLU(),
    nn.Linear(features, 2),
)
# TODO: Change to lymphocyte density recognition path
binary_recognition_model.load_state_dict(
    torch.load(
        "resnet18_binary_lymphocytes.pth",
        map_location="cpu",
    )
)
binary_recognition_model = binary_recognition_model.to(device)
binary_recognition_model.eval()

logger.info("Beginning execution")

save_folder = "results"
img_size_small = (256, 256)
img_size_large = (256, 256)
batch_size_small = 8
batch_size_large = 8
samples = glob.glob(f"{rcc_folder}/**/*.svs")
samples_n = len(samples)
samples = [(sample_idx, img_file) for (sample_idx, img_file) in enumerate(samples)]

# %% Start iteration

with torch.no_grad():
    for sample_idx, img_file in samples:

        dataset_large = SlideDataset(img_file, img_size_large)

        tcga_id = dataset_large.tcga_identifier
        tcga_folder = f"{save_folder}/{tcga_id}"
        os.makedirs(tcga_folder, exist_ok=True)

        classification_exists = os.path.isfile(
            f"{tcga_folder}/tissue_classification.pth"
        )
        if not classification_exists or force_classification:
            logger.info(
                f"{sample_idx + 1:3d}/{samples_n} {tcga_id} Executing texture classification on sample."
            )
            dataset_large.filter_empty(use_cache=True)

            dataloader_large = DataLoader(
                dataset=dataset_large,
                batch_size=batch_size_large,
                pin_memory=True,
                num_workers=cpu_count(),
                shuffle=False,
            )

            classification_preds = torch.zeros(
                [*dataset_large.dimensions] + [classification_class_n]
            )
            classification = torch.zeros(dataset_large.dimensions)
            for index, img_batch in tqdm(
                enumerate(dataloader_large),
                total=int(len(dataset_large) / batch_size_large),
                disable=not verbose,
            ):
                index = min(len(dataset_large), index * batch_size_large)

                positions = dataset_large.image_positions[
                    index : index + batch_size_large
                ]
                positions = [
                    (x // img_size_large[0], y // img_size_large[1])
                    for (x, y) in positions
                ]
                img_batch = img_batch.to(device)

                resp = classification_model(img_batch)
                _, predicted = torch.max(resp, 1)

                for idx, pos in enumerate(positions):
                    classification[pos] = predicted[idx]
                    classification_preds[pos] = resp[idx]
            torch.save(
                classification,
                f"{tcga_folder}/tissue_classification.pth",
            )

            torch.save(
                classification_preds,
                f"{tcga_folder}/tissue_classification_predictions.pth",
            )

            logger.info(
                f"{sample_idx + 1:3d}/{samples_n} {tcga_id} Classification results saved."
            )
        else:
            logger.info(
                f"{sample_idx + 1:3d}/{samples_n} {tcga_id} Texture classification exists."
            )

        binary_recognition_exists = os.path.isfile(
            f"{tcga_folder}/binary_lymphocytes.pth"
        )
        if not binary_recognition_exists or force_binary_recognition:
            logger.info(
                f"{sample_idx + 1:3d}/{samples_n} {tcga_id} Executing binary cell recognition on sample."
            )
            dataset_small = SlideDataset(img_file, img_size_small)
            dataset_small.filter_empty(use_cache=True)
            binary_classification = torch.zeros((*dataset_small.dimensions, 2))

            dataloader_small = DataLoader(
                dataset=dataset_small,
                batch_size=batch_size_small,
                pin_memory=True,
                num_workers=cpu_count(),
                shuffle=False,
            )

            for index, img_batch in tqdm(
                enumerate(dataloader_small),
                total=int(len(dataset_small) / batch_size_small),
                disable=not verbose,
            ):
                index = min(len(dataset_small), index * batch_size_small)

                positions = dataset_small.image_positions[
                    index : index + batch_size_small
                ]
                positions = [
                    (x // img_size_small[0], y // img_size_small[1])
                    for (x, y) in positions
                ]
                img_batch = img_batch.to(device)

                responses = binary_recognition_model(img_batch)
                responses = nn.functional.softmax(responses, dim=1)

                for resp_idx, value in enumerate(responses):

                    pos = positions[resp_idx]
                    binary_classification[pos] = value

            torch.save(
                binary_classification,
                f"{tcga_folder}/binary_lymphocytes.pth",
            )
            logger.info(
                f"{sample_idx + 1:3d}/{samples_n} {tcga_id} Binary recognition results saved."
            )
        else:
            logger.info(
                f"{sample_idx + 1:3d}/{samples_n} {tcga_id} Binary cell recognition exists."
            )

print("Done")
# %%
