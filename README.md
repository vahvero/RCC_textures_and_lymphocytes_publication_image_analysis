# Integrative Analysis of Tissue Textures and Lymphocyte Infiltration in Renal Cell Carcinoma using Deep Learning


## BACKGROUND
These codes will help you reproduce all plots and statistical analyses of the publication "Integrative Analysis of Tissue Textures and Lymphocyte Infiltration in Renal Cell Carcinoma using Deep Learning" by Brummer Otso et al.  


## USEFUL LINKS
**[CODE](https://github.com/obruck/RCC_textures_and_lymphocytes_publication_data_analysis)** &emsp; &emsp; &emsp; &emsp; &nbsp; &nbsp; The codes to reproduce the analyses using image analysis data and TCGA-KIRC clinical, transcriptome and genomic data. To be run after the image analysis pipeline of this repository.  
**[DATA](https://zenodo.org/deposit/6384627)** &emsp; &emsp; &emsp; &emsp; &emsp; The annotated texture and lymphocyte image data are located in Zenodo.  
**[VISUALIZATION](http://hruh-20.it.helsinki.fi/rcc_texture_lymphocytes/)** &emsp; &nbsp;The TissUUmaps platform to visualize the texture and lymphocyte data.


## REQUIREMENTS

```shell
# Create virtual environment
python -m venv .venv # Tested with python3.9.12
source .venv/bin/activate # activate virtual environment
pip install -r requirements.txt
```

## RUNNING

**1. Training**  
The training code `train_network.py` will train models classifying 256x256px image tiles into tissue textures (Cancer, Stroma, Normal, Blood, Empty or Other) or lymphocyte density (Low or High). In addition, the code will produce confusion matrices on the classification accuracy in the test set (10%) and in the entire annotated image dataset (n=39,473 texture images and 25,097 lymphocyte images). Training took ~1 hour with 1 GPU.

Parameters to edit:  
Lines 28-29
```# root = "binary_lymphocytes"
root = "tissue_classification"
```
Comment the row, which you do not want to train. Here, the code will train the tissue texture classifier, but not the lymphocyte classifier.  

Lines 112-120  
```model_name = "resnet18"

models = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
}
```
The `model_name` parameter is used to define the model infrastructure to be trained. Here, "resnet18" is ResNet-18. The alternatives are listed in rows 115-119.  



```shell
nohup python -u train_network.py > cmd_training.log &
```

**2. Inference**  
The training code `inference.py` will run the models on the TCGA-KIRC image dataset and save the texture type and lymphocyte proportion at the tile level. Inference took 2 days with 20x6Gb CPUs.  

Parameters to edit:  

Line 25-26
```
force_classification = False
force_binary_recognition = False
```
If you have already run the inference once, and you need to repeat it only to the texture (force_classification) or lymphocyte (force_binary_recognition) classifier, change the respective classifier to "True". For example, to run inference only for the texture classification, type
```
force_classification = True
force_binary_recognition = False
```  

Line 57
```classification_model = models["resnet18"](```
If you have trained the texture classification model with another infrastructure, replace the `resnet18` with that, for example `resnet34` for ResNet-34.  


Line 76
```binary_recognition_model = models["resnet18"](```
If you have trained the lymphocyte classification model with another infrastructure, replace the `resnet18` with that, for example `resnet34` for ResNet-34.
- Line 102, the path to folder of the WSI `svs`-files is script spesific and must be corrected
- The unique TCGA identifier is expected to be the first 14 letters of the WSI filename. If this is not the case, `utils.extract_tcga_identifier` must be altered to extract the identifer for saving the results.  

After these alterations, the inference can be ran with:  

```sh
nohup python inference.py
```

**3. Export**  
To export the image analysis data to a `raw_data.xlsx` file (495 rows, 78 columns), run
```shell
nohup python -u analysis.py > cmd_analysis.log &
```
