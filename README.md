# Ulcervision 🚑🧠  
**Smart Foot Ulcer Multi-Class Detection using Decision-Level Fusion**

## 🧾 Overview

Ulcervision is a deep learning-based wound classification system designed to assist in the **automated diagnosis of chronic foot ulcers**. It classifies wound images into **six distinct classes**:
- Diabetic
- Pressure
- Venous
- Surgical
- Natural (Healthy)
- Background

Unlike prior research that focused on binary or low-class classification, this project demonstrates the power of **multi-class classification** and **decision-level model fusion**, achieving **state-of-the-art accuracy (89.47%)** on the AZH chronic wound dataset.

## 🧠 Core Features

- ✅ **Multi-class classification** using 4 pre-trained CNNs
- 🔗 **Decision-level fusion**: hard voting, soft voting, max-likelihood, and stacking (XGBoost, Random Forest, Logistic Regression)
- 🧰 Trained on an **augmented dataset of 14,880 images**
- 📈 Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- 📦 Model outputs stored in `.keras` format

## 📁 Project Structure

```
Ulcervision/
│
├── models/                 # Pretrained and fine-tuned models
├── dataset/                # Augmented dataset (not included here)
├── notebooks/              # Jupyter notebooks for training and evaluation
├── fusion/                 # Decision-level fusion logic (voting, stacking, etc.)
├── utils/                  # Helper scripts for preprocessing and evaluation
├── results/                # Evaluation metrics, confusion matrices, ROC curves
├── README.md
└── requirements.txt
```

## 🔍 Methodology

We used four ImageNet pre-trained CNNs for feature extraction:

| Model         | Feature Dim | Strength |
|---------------|-------------|----------|
| VGG19         | 25,088      | Deep, structured filters |
| DenseNet201   | 1,920       | Dense connections, reuse |
| MobileNetV2   | 1,280       | Lightweight, efficient |
| NASNetMobile  | 1,056       | AutoML-designed |

These models' predictions were fused using the following strategies:

- `Soft Voting` (Weighted by validation accuracy)
- `Hard Voting`
- `Maximum Likelihood`
- `Stacking`: Using **XGBoost**, **Random Forest**, and **Logistic Regression**

The **stacking approach with XGBoost** outperformed all others with:

- 📊 **Accuracy**: `89.47%`
- 🎯 **Macro F1-score**: `0.89`

## 📦 Requirements

```bash
Python >= 3.12
TensorFlow >= 2.15
scikit-learn
XGBoost
OpenCV
Matplotlib
NumPy
Pandas
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Clone the repository
2. Place your dataset under the `dataset/` directory
3. Train individual models via `notebooks/train_model.ipynb`
4. Run `fusion/stacking_fusion.py` to combine predictions
5. Evaluate with `notebooks/evaluate_models.ipynb`

## 📊 Results

| Model/Fusion        | Accuracy | F1 (Macro) |
|---------------------|----------|------------|
| VGG19               | 80%      | 0.80       |
| DenseNet201         | 72%      | 0.70       |
| NASNetMobile        | 74%      | 0.74       |
| MobileNetV2         | 72%      | 0.71       |
| Soft Voting         | 83%      | 0.82       |
| Max Likelihood      | 81%      | 0.80       |
| Hard Voting         | 80%      | 0.79       |
| **XGBoost (Stacking)** | **89.47%**  | **0.89**       |

## 📷 Sample Predictions

Coming soon...

## 🧪 Future Improvements

- 🔥 Integration of thermal images for deeper wound insights
- 🗺️ Grad-CAM-based Explainable AI (XAI)
- 💻 Web or mobile-based real-time wound detection app
- 🏥 Wound severity scoring and segmentation support

## 📜 Citation

If you use this work in your research, please cite:

```
Ulcervision: Smart Foot Ulcer Multi-Class Detection using Decision-Level Fusion, NUCES Karachi, 2025.
```

## 🤝 Acknowledgments

- Dataset: [AZH Chronic Wound Dataset](https://github.com/....)  
- Supervisor: Miss Sania Urooj  
- Contributors: Muhammad Mushtaq, Abbas Mustafa, Saleh Shamoon

---

> ⚠️ **Note**: Dataset is not included due to size and license restrictions.
