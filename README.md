# Breast Cancer Classification Project
## Overview
This project focuses on building and evaluating a hybrid deep learning model combining Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) to classify breast cancer histopathological images into benign or malignant categories. The dataset used is the publicly available BreaKHis dataset, which includes histology images captured at various magnifications but 400x magnification level is used for classification in this project.
## Features
**Data Preprocessing**: Techniques such as resizing, augmentation (horizontal/vertical flips, rotations, color jittering), normalization, and handling class imbalances are applied.
**Hybrid Model Architecture**: Combines the feature extraction power of Vision Transformers with the classification capabilities of CNNs, ensuring robust learning.
**Model Training and Fine-Tuning**: Includes hyperparameter tuning, dropout regularization, and the use of pre-trained ViT models for transfer learning.
**Evaluation Metrics**: The model's performance is assessed using accuracy, precision, recall, F1-score, confusion matrices, and training/validation loss plots.
**Visualization**: Offers clear data insights through class distribution analysis, loss/accuracy plots, and confusion matrix visualizations.
# Requirements
**Software and Libraries**
- Python 3.8+
- Libraries:
  - PyTorch
  - Timm (for Vision Transformers)
  - NumPy
  - Keras
  - Pandas
  - Matplotlib
  - Scikit-learn
  - Seaborn
# Hardware
GPU support (e.g., NVIDIA CUDA-enabled GPU) is highly recommended for faster training and evaluation.
## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
# Dataset
The BreakHis dataset is used for this project. It contains breast cancer histopathological images categorized into benign and malignant classes. 
The dataset has been resized and preprocessed for use in the models.

The dataset structure after processed and splitting into train and test:

Train Set: /content/drive/MyDrive/BreakHis/BreaKHis_v1/breast_resized/train
Test Set: /content/drive/MyDrive/BreakHis/BreaKHis_v1/breast_resized/test
# Notebooks
The repository includes the following notebooks:

BreakHis_CNN.ipynb
Implementation and evaluation of a CNN-based model.

BreakHis_Preprocess.ipynb
Code for preprocessing the BreakHis dataset (e.g., resizing and augmentation).

BreakHis_VGG16_Updated.ipynb
Updated version of the VGG16 implementation, fine-tuned for better performance.

BreakHis_Vgg16.ipynb
Initial implementation of breast cancer detection using the VGG16 model.

BreakHis_Vit_CNN.ipynb
Implementation of a hybrid model combining Vision Transformer (ViT) and CNN for feature extraction and classification.

README.md
The documentation for the repository.

# Key Sections
Data Exploration: Analyze the dataset, identify class distributions, and apply augmentation to address imbalances.
Model Architecture: Leverage Vision Transformers for feature extraction and CNNs for classification.
Training and Optimization: Perform training with advanced optimization techniques and monitor validation performance.
Performance Evaluation: Generate metrics such as confusion matrices, accuracy, precision, recall, and F1-scores, along with training/validation plots.
# Results
The proposed hybrid ViT-CNN model demonstrated significant performance improvements, achieving high accuracy and robust classification across different magnifications of histopathological images. Results and visualizations are documented in the notebook.
# Future Work
Extend the model to classify additional breast cancer subtypes.
Test the model's generalization on other histopathological datasets.
Explore explainable AI techniques to provide insights into model predictions.
Feel free to contribute, suggest improvements, or report issues
## Contributions
Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## Contact
For questions or issues, please contact nasim.shazia1@gmail.com.
