# 🔬 Breast Cancer Classification — Hybrid ViT-CNN Deep Learning Model

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Model](https://img.shields.io/badge/Model-ViT%20%2B%20CNN%20Hybrid-red)
![Dataset](https://img.shields.io/badge/Dataset-BreakHis%20400x-purple)
![Publication](https://img.shields.io/badge/Published-IEEE%20Access-green)

## 📌 Project Overview

This project develops and evaluates a **novel hybrid deep learning architecture** 
combining **Vision Transformers (ViT)** and **Convolutional Neural Networks (CNN)** 
to classify breast cancer histopathological images as **benign or malignant**.

The **BreakHis dataset** (400× magnification) is used, with full preprocessing, 
augmentation, class imbalance handling, and rigorous multi-metric evaluation.

This research is linked to peer-reviewed publications in **IEEE Access (2022, 2024)**.

---

## 🗂️ Repository Structure

| Notebook | Description |
|---|---|
| `BreakHis_Preprocess.ipynb` | Data loading, resizing, augmentation, class imbalance handling |
| `BreakHis_CNN.ipynb` | Baseline custom CNN — training and evaluation |
| `BreakHis_Vgg16.ipynb` | VGG16 transfer learning — initial implementation |
| `BreakHis_VGG16_Updated.ipynb` | VGG16 fine-tuned with improved optimisation |
| `BreakHis_Vit_CNN.ipynb` | **Hybrid ViT-CNN model** — novel architecture (main contribution) |

---

## 🛠️ Tools & Technologies

- **Language:** Python 3.8+
- **Frameworks:** PyTorch, Keras
- **Libraries:** Timm (ViT), NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Architectures:** Custom CNN, VGG16 (Transfer Learning), Vision Transformer (ViT), Hybrid ViT-CNN
- **Dataset:** BreakHis — Breast Cancer Histopathological Image Dataset (400× magnification)
- **Hardware:** NVIDIA CUDA GPU recommended (Google Colab supported)

---

## 🔍 Methodology

### 1️⃣ Data Preprocessing (`BreakHis_Preprocess.ipynb`)
- Resized histology images to standard input dimensions
- Applied **data augmentation**: horizontal/vertical flips, rotations, colour jittering
- Handled **class imbalance** via weighted sampling strategies
- Normalised pixel values for model input
- Split into stratified train / validation / test sets

### 2️⃣ Baseline CNN (`BreakHis_CNN.ipynb`)
- Custom convolutional architecture with Conv2D, MaxPooling, and Dropout layers
- Trained using Adam optimiser with binary cross-entropy loss
- Established performance baseline for comparison

### 3️⃣ VGG16 Transfer Learning (`BreakHis_Vgg16.ipynb` + `BreakHis_VGG16_Updated.ipynb`)
- Loaded pre-trained VGG16 with ImageNet weights
- Froze base layers → trained custom classification head
- Fine-tuned top layers with learning rate scheduling and early stopping

### 4️⃣ Hybrid ViT-CNN Model (`BreakHis_Vit_CNN.ipynb`)
- **ViT branch:** patch embedding → multi-head self-attention → transformer encoder (global features)
- **CNN branch:** convolutional layers for local spatial feature extraction
- Merged feature representations → dense classification head
- Applied dropout regularisation and hyperparameter tuning
- Achieved best performance across all evaluation metrics

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Custom CNN | ~85% | ~84% | ~83% | ~84% |
| VGG16 (Transfer) | ~89% | ~88% | ~87% | ~88% |
| VGG16 (Fine-tuned) | ~91% | ~90% | ~90% | ~90% |
| **Hybrid ViT-CNN** | **~93%** | **~92%** | **~93%** | **~92%** |

> Full results, confusion matrices, and training/validation plots are documented in each notebook.

---

## ▶️ Installation & How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ShaziaNasim606/BreakHis-Project.git
cd BreakHis-Project

# 2. Install dependencies
pip install torch torchvision timm numpy pandas matplotlib seaborn scikit-learn keras opencv-python

# 3. Download the BreakHis dataset
# https://www.kaggle.com/datasets/ambarish/breakhis
# Place images in ./data/ folder

# 4. Run notebooks in order:
# BreakHis_Preprocess.ipynb  →  BreakHis_CNN.ipynb
# BreakHis_Vgg16.ipynb       →  BreakHis_VGG16_Updated.ipynb
# BreakHis_Vit_CNN.ipynb
```

> 💡 **Recommended:** Run on Google Colab with GPU runtime for faster training.

---

## 📈 Key Features

- **Data Exploration:** Class distribution analysis with augmentation to address imbalance
- **Hybrid Architecture:** ViT global attention + CNN local feature extraction combined
- **Training & Optimisation:** Hyperparameter tuning, dropout regularisation, early stopping, learning rate scheduling
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Loss/Accuracy plots
- **Visualisation:** Training curves, confusion matrices, class distribution charts

---

## 🔭 Future Work

- Extend classification to additional breast cancer subtypes beyond benign/malignant
- Test model generalisation on other histopathological datasets
- Integrate **explainable AI (XAI)** techniques (Grad-CAM, SHAP) to visualise model decisions
- Explore multi-magnification training (40×, 100×, 200×, 400× combined)

---

## 🤝 Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## 👩‍💻 Author

**Shazia Nasim**
MSc Data Science | University of Hertfordshire
📍 Bristol, UK | 📧 nasim.shazia1@gmail.com
🔗 [GitHub Profile](https://github.com/ShaziaNasim606)
