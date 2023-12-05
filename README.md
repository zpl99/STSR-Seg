# STSR-Seg

The code of the Spatiotemporal aware Super-Resolution SEGmentation framework (STSR-Seg). The STSR-Seg is utilized to produce the China Building Rooftop Area (CBRA) dataset. 

### Requirements
1. Use conda to create vitrual environment and activate it
```
conda create -n [yourenvname] python=3.8
conda activate [yourenvname] 
```
2. Install PyTorch=1.7 following the [official instructions](https://pytorch.org/), install earth-engine following the [official instructions](https://developers.google.com/earth-engine/guides/python_install). For users in China mainland, you can refer to Zhihu to install earth-engine and solve the problems that may occur during the installation process [Zhihu](https://zhuanlan.zhihu.com/p/29186942)
3. Install dependencies: pip install -r requirements.txt
4. Download the [pretrained Resnet-50 backbone](https://drive.google.com/file/d/1EZFEiqcMiSPDqtXOjfgKnjGsQAW0IpoD/view?usp=sharing), and put it in the "Pretrained_models" folder,  
```
Project path
  |-- Dataset
  |-- Download
      ......
  |-- Pretrained_models
      |--resnet50-deep.pth
  main.py
  ......
```
### Step 1: prepare the training data and the validation data
1. Run the following command to download low-resolution training data and save it to your specified path, e.g., [your lr save path]
```
python Download/s2andDynamicWorld_download.py --save_path [your lr save path]
```
2. Run the following command to prepare the low-resolution training data, then you can find a CSV file labelled "lrFiles.csv" in path ./Misc
```
python Misc/prepareLR_LR.py --data_path [your lr save path]
```
3. Download high-resolution training data from [google drive](https://drive.google.com/file/d/1VUY2NTJDDa-Byjue41lyhp7ExEvY90Bd/view?usp=sharing) and unzip it to your specified path, e.g., [your hr save path]
### Step 2: train the model
1. Open config.py, edit the following entry and save it
```
__C.DATASET.VAL = "[your hr save path]/valData"
__C.DATASET.LR_HR = "[your hr save path]/trainData"
__C.DATASET.LR_LR = "Misc/lrFiles.csv"
```
2. Run the following code to train the model
```
python main.py
```
You can see the help of the main script with
```
python main.py --help
```
TC loss is more demanding on computational resources and can also be turned off, if you choose to turn it off you can run the following code for model training
```
python main.py --framework lr_lr_and_lr_hr --dataset MultiData
```
After training, you can find several checkpoints in your project path, for example:
```
Project path
  |-- MultiDataEDSRUnet-model-10.ckpt
  |-- MultiDataEDSRUnet-model-20.ckpt
  |-- ......
  |-- MultiDataEDSRUnet-model-best.ckpt
```
### Step 3: inference to achieve building rooftop extraction
1. Download Sentinel-2 data and the corresponding Dynamic World data (built), save them under [your s2 path] and [your dynamicworld path], making sure that the name of the Sentinel-2 image matches the name of the Dynamic World data. For example, you can prepare your data as the following structure
```
[your s2 path]
    |-- China0001.tif
    |-- China0002.tif
    ......
[your dynamicworld path]
    |-- China0001.tif
    |-- China0002.tif
    ......
```
2. Run the following code. [your save path] is where you would like to put your predictions, [your temporary files path] is the path where the temporary file is saved, [your checkpoint path] is the path of the checkpoint you would like to use.
```
python inference.py --s2Path [your s2 path] --luccPath [your dynamicworld path] --desPath [your save path] --tempPath [your temporary files path] --checkpointPath [your checkpoint path]
```
### Citation
Liu, Z., Tang, H., Feng, L., & Lyu, S. (2023). China Building Rooftop Area: the first multi-annual (2016–2021) and high-resolution (2.5 m) building rooftop area dataset in China derived with super-resolution segmentation from Sentinel-2 imagery. Earth System Science Data, 15(8), 3547-3572. DOI: 10.5194/essd-15-3547-2023.


