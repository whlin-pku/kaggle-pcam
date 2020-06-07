# PatchCamelyon

### Introduction

- **Problem:** The goal is to create an algorithm to **identify metastatic cancer** in small image patches taken from larger digital pathology scan.
- **Dataset:** The PatchCamelyon benchmark (**Pcam**), derived from the Camelyon16 Challenge,  has following features:
  - type: HDF5 File
  - input: 96x96
  - train/valid/test: 262144 / 32768 / 32768 
  - label: binary (positive 1 / negative 0)
- **Our Approach：**
  - Using **transfer learning** with ResNet-50 as our backbone (pretrained by imagenet)
  - **Discriminative learning rate** to finetune
  - Using **Adam** as optimizer and **Cross Entropy** as our loss function
  - Using **Learning rate decay** to achieve better convergence
  - Replace ReLU() activation with **SELU()** in fully connected layer.
  - To alleviate the over-fitting problem, we use **Data Augmentation ( > 10), Weight Decay, Dropout**.

### Usage

```python
# recommand using anaconda to create virtual environment
conda create -n pcam python=3.7
conda activate pcam

# install package through tsinghua channel
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# run train/test, single-gpu version
python train.py
python test.py
```

### Dataset Downloads

[**basveeling/pcam**](https://github.com/basveeling/pcam)