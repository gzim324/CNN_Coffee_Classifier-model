# CNN Coffee Classifier

The repository consists of three parts of experiments:  

- **custom_training** – training custom CNN models with a self-defined architecture.  
- **experiments** – training prebuilt models (MobileNet, ResNet50, ResNet101) from scratch, without ImageNet weights.  
- **cv10_experiments** – training models (MobileNet, ResNet50, ResNet101) with ImageNet fine-tuning and 10-fold cross-validation; additionally, 10-fold cross-validation for 3 custom models (Adam, Adadelta, SGD).  

---

## Dataset  

Download the dataset from [https://example.com/dataset](https://example.com/dataset) and place it into the following directories:  

- `custom_training/data`  
- `experiments/data`  
- `cv10_experiments/data-ar`  

---


## Setup and run

Install miniconda from https://www.anaconda.com/docs/getting-started/miniconda/main

---

Create environment and install packages
```bash
 conda create -n coffee-env python=3.9 -y
 conda activate coffee-env
 
 pip install --upgrade pip
 pip install -r requirements.txt
```

Custom architecture
```bash
 cd custom_training
 python train_model.py optimizer=SGD dropout=0.3 kernel_size=3 batch_size=32
```

Using mobilenet, resnet50 and resnet101
```bash
 cd experiments
 python mobilenet_ft.py
 python mobilenet.py
 python train_resnet101_ft.py
 python train_resnet101.py
 python train_resnet50_ft.py
 python train_resnet50.py
```

10 cross validation
```bash
 cd cv10_experiments
 python train_custom_adam.py
 python train_custom_sgd.py
 python train_mobilenet_ft.py
 python train_mobilenet.py
 python train_resnet101_ft.py
 python train_resnet101.py
 python train_resnet50_ft.py
 python train_resnet50.py
```
