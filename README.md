# 🗑️ Waste Classification Using Deep Learning  

Implementation of **RWC-Net** (DenseNet201 + MobileNetV2 hybrid) for **Recyclable Waste Classification**.  
"A Reliable and Robust Deep Learning Model for Effective Recyclable Waste Classification".  

## 📌 Project Overview
This project explores **deep learning architectures** for image classification tasks using PyTorch.  
We benchmark standard models (**DenseNet201, MobileNetV2**) against our **proposed RWC-Net** architecture.  
The goal is to evaluate performance improvements through custom design and optimization.

---

## 🗂️ Features
- ✅ Data preprocessing and augmentation with `torchvision.transforms`
- ✅ Training and evaluation pipeline for multiple models
- ✅ Custom architecture: **RWC-Net**
- ✅ Performance metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- ✅ Visualization utilities: Confusion Matrix, ROC curve, PR curve

---
## 📥 Dataset
We use the [**TrashNet dataset**](https://github.com/garythung/trashnet), which contains **2527 images** across **6 waste categories**:  
- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash (Litter)  

Place the dataset inside:
```
data/trashnet/
```

Expected folder structure:
```
data/trashnet/
│── cardboard/
│── glass/
│── metal/
│── paper/
│── plastic/
│── trash/
```

---

## 🛠️ Preprocessing
Defined in `src/dataset.py`:  
- **Resize**: `224 × 224`  
- **Augmentation**:  
  - Random Horizontal Flip (p=0.5)  
  - Random Rotation (±30°)  
  - Random Crop (224×224)  
- **Normalization**: ImageNet mean & std  
  - Mean = `[0.485, 0.456, 0.406]`  
  - Std = `[0.229, 0.224, 0.225]`  

---

## 🤖 Model: RWC-Net
The **RWC-Net** model fuses two pretrained CNN backbones:  
- **DenseNet201**  
- **MobileNetV2**  

### Features:
- Extract embeddings from both backbones  
- Concatenate → fully connected fusion layer  
- Auxiliary outputs from DenseNet and MobileNet  
- Final classifier with **LogSoftmax**  

Run training:
```bash
python src/model.py
```
---

## ⚙️ Training
We use **5-Fold Cross Validation** to ensure robustness.  

**Training setup** (`src/training.py`):  
- Optimizer: **Adam** (`lr = 1e-5`)  
- Loss: **CrossEntropyLoss** with weighted auxiliary losses  
  - Main loss: `1.0`  
  - Aux1: `0.5`  
  - Aux2: `0.25`  
- Evaluation: **Accuracy, Precision, Recall, F1-score**  

Run training:
```bash
python src/training.py
```
---

## 📊 Evaluation
Models are evaluated using **5-Fold Cross Validation** with the following metrics:  
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC  

We also generate plots: **Confusion Matrix, ROC Curve, Precision–Recall Curve, and Training vs Validation curves**.  
Results are averaged across folds for fair comparison.  

Run training:
```bash
python src/evaluation.py
```
---

## 📊 Models Compared
1. **DenseNet201 (Baseline)**
2. **MobileNetV2 (Baseline)**
3. **RWC-Net (Proposed Model)**

---

## ⚡ Setup
Clone the repo and install dependencies:
```bash
git clone https://github.com/vinay2222222/Waste-Classification-using-Deep-Learning.git
cd Waste-Classification-using-Deep-Learning
pip install -r requirements.txt
```

---

## 📈 Results
- **DenseNet201**: Baseline accuracy XX%  
- **MobileNetV2**: Baseline accuracy XX%  
- **RWC-Net**: Achieved **higher accuracy of XX%** with improved F1-score.  

*(Replace XX% with your actual results from notebook)*

---

## 📂 Repository Structure
```
RWC-Net-Waste-Classification/
│── README.md
│── requirements.txt
│── .gitignore
│
├── data/
│   └── trashnet/                   # Dataset
│
├── notebooks/
│   ├── RWCNet_Preprocessing.ipynb  # Colab notebook (dataset + preprocessing)
│   └── RWCNet_Training.ipynb       # Colab notebook (model training)
|   └── final_model.ipynb           # Colab notebook (dataset + preprocessing + model training + evaluation)
│
└── src/
    ├── dataset.py                  # Dataset loading, preprocessing, augmentation
    ├── model.py                    # RWC-Net
    └──training.py                  # Training of model
    └──evaluation.py                # Evaluation
```

---

## 🔮 Future Work
- Extend to larger datasets (e.g., CIFAR-100, ImageNet subset)
- Optimize hyperparameters for RWC-Net
- Deploy trained model as a web app (Streamlit/Flask)

---

## 👨‍💻 Authors
- Vinay A ([@vinay2222222](https://github.com/vinay2222222))  
