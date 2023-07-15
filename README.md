# STSR-Seg

The code of the Spatio-Temporal aware Super-Resolution SEGmentation framework (STSR-Seg). The STSR-Seg is utilized to produce the China Building Rooftop Area (CBRA) dataset. 

The page is still being organized, the paper related to the method is being reviewed by the journal "Earth System Science Data (ESSD)", and we will release all the source code in the near future.

### Requirements
1. Please install PyTorch=1.7 following the [official instructions](https://pytorch.org/), install earth-engine following the [official instructions](https://developers.google.com/earth-engine/guides/python_install). For users in China mainland, you can refer to Zhihu to install earth-engine and solve the problems that may occur during the installation process [Zhihu](https://zhuanlan.zhihu.com/p/29186942)
2. Install dependencies: pip install -r requirements.txt
### Step 1: prepare the training data
1. Run the following command to download low-resolution training data and save it to your specified path, e.g., [your lr save path]
```
python Download/s2andDynamicWorld_download.py --save_path [your lr save path]
```
2. Run the following command to prepare the low-resolution training data, and you can find a CSV file labelled "lrFiles.csv" in path ./Misc
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
2. Run the following code
```
python main.py
```
TC loss is more demanding on computational resources and can also be turned off, if you choose to turn it off you can run the following code
```
python main.py --framework lr_lr_and_lr_hr --dataset MultiData
```
After training, you can find several checkpoints in your project path:
```
Project path
  ├── MultiData_wTCEDSRUnet-model-10.ckpt
  ├── MultiData_wTCEDSRUnet-model-20.ckpt
  ├── ......
  ├── MultiData_wTCEDSRUnet-model-best.ckpt
```
### Step 3: inference to achieve building rooftop extraction
1. Download Sentinel-2 data and the corresponding Dynamic World data (built), save them under [your s2 path] and [your dynamicworld path], making sure that the name of the Sentinel-2 image matches the name of the Dynamic World data. For example, you can prepare your data as the following structure
```
[your s2 path]
    ├── China0001.tif
    ├── China0002.tif
    ......
[your dynamicworld path]
    ├── China0001.tif
    ├── China0002.tif
    ......
```
2. Run the following code. [your save path] is where you would like to put your predictions, and [your temporary files path] is the path where the temporary file is saved.
```
python inference.py --s2Path [your s2 path] --luccPath [your dynamicworld path] --desPath [your save path] --tempPath [your temporary files path]
```
