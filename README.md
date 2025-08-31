# 🗑️ Waste Classification Using Deep Learning

Implementation of **RWC-Net** (DenseNet201 + MobileNetV2 hybrid) for **Recyclable Waste Classification**.
**"A Reliable and Robust Deep Learning Model for Effective Recyclable Waste Classification"**.  

This repo currently includes dataset setup and **preprocessing pipeline** (augmentation + normalization).  

---

## 📂 Repository Structure
```
RWC-Net-Waste-Classification/
│── README.md
│── requirements.txt
│── .gitignore
│
├── data/
│   └── trashnet/            # Placeholder for dataset
│
├── notebooks/
│   └── RWCNet_Preprocessing.ipynb   # Colab notebook (dataset + preprocessing)
│
└── src/
    ├── dataset.py           # Dataset loading, preprocessing, augmentation
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

Download and place the dataset inside:
```
data/trashnet/
```

The dataset folder structure should look like:
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
We apply the following preprocessing steps (defined in `src/dataset.py`):  
- **Resize** images to `224 × 224`  
- **Data Augmentation**:  
  - Random Horizontal Flip (p=0.5)  
  - Random Rotation (±30°)  
  - Random Crop (224×224)  
- **Normalization**: ImageNet mean & std  
  - Mean = `[0.485, 0.456, 0.406]`  
  - Std = `[0.229, 0.224, 0.225]`  

---

## ▶️ Usage
Run the preprocessing pipeline to get train/validation/test splits:  

```bash
python src/dataset.py
```

## 📒 Notebook Demo
For interactive testing, open the Colab notebook:  
👉 `notebooks/Preprocessing.ipynb`  

It will:  
1. Download TrashNet.  
2. Apply preprocessing pipeline.  
3. Visualize augmented samples.  

---

## 📌 Next Steps
- [ ] Add **RWC-Net model (DenseNet201 + MobileNetV2 fusion)**  
- [ ] Implement **5-fold cross-validation training**  
- [ ] Add **Score-CAM visualization** for interpretability  

---

✍️ Maintainer: Vinay A  
