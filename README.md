# Quantification of lymphocytes and tissue textures from KIRC-TCGA samples using computational histopathology

## Requirements

```shell
# Create virtual environment
python -m venv .venv # Tested with python3.9.12
source .venv/bin/activate # activate virtual environment
pip install -r requirements.txt
```

## Running

Some parameters spesific to the script `interference.py` are required be changed:

- On line 59, the path to folder of the WSI `svs`-files is script spesific and must be corrected
- On line 72, if it is not required to run interference over all samples, a reduction to the payload is highly suggested, as the computational requirements are high for each sample.
- The unique TCGA identifier is expected to be the first 14 letters of the WSI filename. If this is not the case, `utils.extract_tcga_identifier` must be altered to extract the identifer for saving the results.

After these alterations, the interference can be ran with:

```sh
python interference.py
```

