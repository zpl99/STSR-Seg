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
3. Download high-resolution training data from [google drive]() and unzip it to your specified path, e.g., [your hr save path]
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
### Step 3: inference to achieve large-scale building rooftop extraction
