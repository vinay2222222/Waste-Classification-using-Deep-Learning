# 🗑️ Waste Classification Using Deep Learning  

Implementation of **RWC-Net** (DenseNet201 + MobileNetV2 hybrid) for **Recyclable Waste Classification**.  
"A Reliable and Robust Deep Learning Model for Effective Recyclable Waste Classification".  

This repo includes:  
✔️ Dataset setup & preprocessing  
✔️ **RWC-Net model (DenseNet201 + MobileNetV2 fusion)**  
✔️ **5-fold cross-validation training pipeline**  

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
│   └── RWCNet_Training.ipynb       # Colab notebook (model training + CV)
│
└── src/
    ├── dataset.py                  # Dataset loading, preprocessing, augmentation
    ├── rwcnet_training.py          # RWC-Net + 5-fold CV training
```

---

## ⚡ Setup
Clone the repo and install dependencies:
```bash
git clone https://github.com/vinay2222222/Waste-Classification-using-Deep-Learning.git
cd Waste-Classification-using-Deep-Learning
pip install -r requirements.txt
```

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

## 📒 Notebooks
- `Preprocessing.ipynb` → Dataset setup + augmentations visualization  
- `Training.ipynb` → Full training with 5-fold CV (Colab-ready)  

---

## 📊 Sample Training Logs
Example output during training (1 fold, 5 epochs):

➡️ After 5 folds, average accuracy is around **83–85%**, depending on augmentation randomness.  

---

## 📌 Next Steps (Roadmap)

### 🔹 1. Model Benchmarking (5 Models Compared)
- **ResNet50** (baseline)  
- **DenseNet121** (baseline)  
- **MobileNetV2** (baseline)  
- **EfficientNet-style** (baseline)  
- **RWC-Net (Proposed Model)**  

➡️ Compare Accuracy, Precision, Recall, F1-score across models.  

---

### 🔹 2. Training & Evaluation Pipeline
- [ ] Automated training for all models  
- [ ] Detailed evaluation metrics  
- [ ] Visualization dashboard for results  
- [ ] Model checkpointing & saving  

---

### 🔹 3. Deployment Options
**A. Gradio Web Interface**  
- Drag-and-drop waste image upload  
- Real-time classification with confidence scores  
- Clean UI with probabilities  
- Shareable public demo  

**B. REST API (Flask)**  
- JSON-based prediction endpoint  
- Health check endpoint  
- Easy system integration  

**C. Batch Processing**  
- Predict multiple images from a folder  
- Export CSV with predictions  
- Error handling for invalid images  


✍️ Maintainer: **Vinay A**  
