# 🦷 ML vs DL — Dental Image Classification

A comparative study between classical Machine Learning and Deep Learning approaches for classifying dental conditions using the Teeth Dataset.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ResNet50 (Transfer + Fine-Tuning) | **99.51%** | **99.52%** | **99.51%** | **99.51%** |
| MobileNetV2 (Transfer Learning) | 97.08% | 97.21% | 97.08% | 97.08% |
| SVM (HOG + RBF) | 63.04% | 63.64% | 63.04% | 62.80% |
| KNN (K=5, HOG + PCA) | 58.56% | 65.99% | 58.56% | 57.90% |

---

## 🗂️ Dataset

- **Classes (7):** CaS · CoS · Gum · MC · OC · OLP · OT
- **Train:** 3,087 images · **Validation:** 1,028 images · **Test:** 1,028 images

---

## 🔵 Machine Learning Models

### SVM — Support Vector Machine
- Feature extraction: **HOG** (Histogram of Oriented Gradients) on 64×64 grayscale images
- Normalization: **StandardScaler**
- Kernel: **RBF** · C=10 · gamma='scale'

### KNN — K-Nearest Neighbors
- Feature extraction: **HOG** → **PCA** (95% variance retained)
- K=5 · weights='distance' · metric='euclidean'

---

## 🟠 Deep Learning Models

### MobileNetV2 — Transfer Learning
- Base: MobileNetV2 pretrained on ImageNet (frozen)
- Head: GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)
- Optimizer: Adam · Epochs: 20

### ResNet50 — Transfer Learning + Fine-Tuning
- **Phase 1:** Base frozen, train custom head · Epochs: 20
- **Phase 2:** Unfreeze last 30 layers · LR: 1e-5 · Epochs: 10
- Head: GlobalAveragePooling → BatchNorm → Dense(256, ReLU) → Dropout(0.3) → Dense(7, Softmax)

---

## 🌐 Live Demo

👉 **[Try the app on Streamlit](https://ml-vs-dl-image-classification.streamlit.app/)**

Upload a dental image and get an instant prediction powered by ResNet50.

---

## 📁 Repository Structure

```
├── app.py                               # Streamlit web app
├── requirements.txt                     # Python dependencies
├── ML_vs_DL_Teeth_Classification.ipynb  # Full experiment notebook
└── README.md
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/itsYoussefAI/ml-vs-dl-image-classification.git
cd ml-vs-dl-image-classification
pip install -r requirements.txt
streamlit run app.py
```
