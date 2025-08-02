# Machine-Generated vs Human-Written Text Detection

**Author:** Thep Rungpholsatit  
**Student Number:** 041066248  
**Course:** CST8507 - Natural Language Processing  
**Date:** Aug 2, 2025  

## 📝 Abstract

This project tackles the growing challenge of distinguishing between human-written and AI-generated text. A binary classifier was built using the English subset of the SemEval-2024 Task 8 dataset. Two approaches were evaluated:

- TF-IDF with Logistic Regression
- Fine-tuned BERT (transformer-based)

Results show BERT significantly outperforms traditional models in accuracy and generalization, demonstrating the effectiveness of transformer-based methods in detecting machine-generated content.

---

## 1. 📌 Introduction

With the rise of large language models like GPT-3 and ChatGPT, distinguishing between human and machine-generated text has become difficult—especially in education, journalism, and online moderation.

We built a binary classifier using two approaches:
- TF-IDF + Logistic Regression
- Fine-tuned BERT

**Research Question:**  
Can machine learning techniques accurately detect whether a given text is human-written or machine-generated?

---

## 2. 📚 Dataset

**Source:** [SemEval-2024 Task 8](https://github.com/mbzuai-nlp/SemEval2024-task8)

- English subset used.
- Each file labeled: `human` or `machine`.
- Varying sentence lengths.

**Preprocessing:**
- Removed noise & short texts
- Balanced classes
- Lowercased text
- Removed special characters & stopwords
- Stratified train/validation/test split

---

## 3. 🛠️ Method

### 3.1 Baseline Models (TF-IDF + ML)
- TF-IDF vectorization
- Models used:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM

These models assume writing origin is reflected in word frequency patterns.

### 3.2 Transformer-Based Model (BERT)
- Fine-tuned `bert-base-uncased`
- Used BERT tokenizer
- Trained for 4 epochs
- Optimizer: AdamW
- Loss: Binary cross-entropy

BERT captures deeper contextual features for nuanced detection.

---

## 4. 📊 Results

### Baseline Models
Each model trained using TF-IDF:
- Logistic Regression
- Naive Bayes
- Linear SVM

### Transformer Model
- Fine-tuned BERT on full monolingual dataset

BERT showed significantly higher accuracy and better generalization.

---

## 5. 🚧 Challenges and Solutions

- **Challenge:** BERT fine-tuning was slow on CPU  
  **Solution:** Reduced epochs and optimized batch size.

- **Challenge:** Dataset imbalance in baseline models  
  **Solution:** Downsampled to balance classes.

---

## 6. 💡 Discussion and Future Work

**Findings:**
- BERT outperformed traditional models in detecting AI-generated text.
- Frequency-based models lack contextual understanding.

**Future Directions:**
- Multi-language support
- Adversarial robustness
- Zero-shot/few-shot with larger LLMs (e.g., GPT-4)

---

## 7. 📚 References

- Al-Khatib et al. (2024) - SemEval-2024 Task 8  
  https://github.com/mbzuai-nlp/SemEval2024-task8  
- Devlin et al. (2019) - BERT  
  https://doi.org/10.48550/arXiv.1810.04805  
- Pedregosa et al. (2011) - scikit-learn  
  https://jmlr.org/papers/v12/pedregosa11a.html  
- Wolf et al. (2020) - Huggingface Transformers  
  https://doi.org/10.18653/v1/2020.emnlp-demos.6

---

## 📁 Repository Structure

```
project_nlp/
├── data/
│   └── SemEval2024/
├── notebooks/
│   ├── 01_tfidf_baselines.ipynb
│   └── 02_finetune_bert.ipynb
├── distilbert_trained_model.pt
└── README.md
```

---

## 🧪 How to Run

```bash
# Step 1: Create virtual environment
python3 -m venv project_nlp
source project_nlp/bin/activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run notebooks
jupyter notebook
```

---

> ✅ For reproducibility, refer to scripts in `/notebooks` and pre-trained models provided.
