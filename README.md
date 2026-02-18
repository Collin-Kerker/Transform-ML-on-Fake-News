# Fake News Detection with BERT Transformers

A natural language processing project using BERT (Bidirectional Encoder Representations from Transformers) to classify news articles as real or fake with 99.97% accuracy.

## Project Overview

This project tackles the critical problem of misinformation detection by building a machine learning model that can automatically classify news articles as either real or fake. With the rapid spread of false information online, automated fact-checking tools are increasingly important for maintaining information integrity.

The model uses a fine-tuned **BERT transformer** trained on approximately 44,000 labeled news articles. BERT's bidirectional context understanding makes it particularly well-suited for analyzing complex news articles with nuanced language and deep contextual meaning.

## Why This Matters

The unfiltered nature of online publishing means false information can spread at alarming rates. This project demonstrates:
- How transformer models can effectively identify misinformation
- The potential for automated fact-checking systems
- The importance of AI in combating the spread of fake news

## Dataset

**Source:** [Fake-and-Real-News-Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by clmentbisaillon on Kaggle

**Dataset Details:**
- **Total Articles:** ~44,000 news articles
- **Real News:** ~21,000 articles from Reuters.com
- **Fake News:** ~23,000 articles from various unreliable sources
- **Features:** Article title and full text content
- **Labels:** Binary (0 = Fake, 1 = Real)

**Data Preprocessing:**
- Combined title and text into single feature
- Removed duplicates and null values
- Shuffled dataset for unbiased training
- Train/Test split: 80/20

## Model Architecture

### BERT Transformer
- **Base Model:** `bert-base-uncased` from Hugging Face
- **Task:** Binary sequence classification
- **Tokenization:** BERT tokenizer with max length of 512 tokens
- **Training:** Fine-tuned on news article classification task

### Baseline Model (For Comparison)
- **Approach:** TF-IDF vectorization + Logistic Regression
- **Purpose:** Establish performance benchmark

## Results

### Performance Metrics

| Model | Accuracy | F1 Score | 
|-------|----------|----------|
| **BERT Transformer** | **99.97%** | **99.98%** |
| Baseline (TF-IDF + LogReg) | 98.64% | 98.77% |

### Confusion Matrix Results

**BERT Performance on Test Set:**
- **True Negatives (Fake → Fake):** 3,495
- **False Positives (Fake → Real):** 0
- **True Positives (Real → Real):** 4,233
- **False Negatives (Real → Fake):** 2

**Key Findings:**
- BERT achieved near-perfect classification with only 2 errors out of 7,730 test samples
- The model perfectly identified all fake news articles (0 false positives)
- Only 2 real news articles were misclassified as fake (showing slight conservative bias)
- BERT outperformed the strong baseline by 1.3 percentage points in accuracy

## Technologies Used

**Language & Environment:**
- Python 3.11
- Jupyter Notebook

**Core Libraries:**
- `transformers` - Hugging Face library for BERT implementation
- `datasets` - Dataset loading and processing
- `torch` - PyTorch deep learning framework
- `accelerate` - Distributed training support
- `scikit-learn` - Baseline model and metrics
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` & `seaborn` - Visualization

## Installation & Setup

### Prerequisites
- Python 3.11 (or 3.8+)
- pip package manager
- ~4GB free disk space for model downloads

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/BetterButterBoy/422-Final-Project.git
cd 422-Final-Project
```

2. **Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn transformers datasets accelerate torch
```

Alternatively, if using a `requirements.txt`:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Download the [Fake-and-Real-News-Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) from Kaggle
- Place `Fake.csv` and `True.csv` in the `data/` folder

### Required Dependencies
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.15.0
torch>=1.10.0
```

## Running the Project

### Jupyter Notebook (Recommended)
```bash
jupyter notebook Transformer_ML_Notebook.ipynb
```

Run all cells sequentially to:
1. Load and preprocess the dataset
2. Create baseline model
3. Fine-tune BERT transformer
4. Evaluate both models
5. Generate confusion matrices and comparison metrics

### Training Details
- **Training Time:** Varies by hardware (CPU: ~2-4 hours, GPU: ~30-60 minutes)
- **Hardware Used:** CPU training (GPU recommended for faster results)
- **Batch Size:** Default settings from Hugging Face Trainer
- **Epochs:** Configured for optimal performance

## Model Pipeline

```
Raw Data (CSV files)
    ↓
Data Cleaning & Combination
    ↓
Train/Test Split (80/20)
    ↓
BERT Tokenization (max_length=512)
    ↓
Fine-tuning BERT Model
    ↓
Evaluation & Metrics
    ↓
Confusion Matrix Analysis
```

## Key Learnings

### Technical Insights
- **BERT's Effectiveness:** Transformer models excel at understanding context in text classification tasks
- **Dataset Characteristics:** The clear linguistic differences between fake and real news made this dataset well-suited for classification
- **Baseline Strength:** Even traditional methods (TF-IDF + Logistic Regression) achieved 98.6% accuracy, indicating distinctive patterns in fake news writing

### Practical Applications
- High accuracy makes this approach viable for real-world misinformation detection
- Model shows conservative bias (2 false negatives, 0 false positives), which is preferable for fact-checking
- Demonstrates potential for automated content moderation systems

## Project Structure

```
422-Final-Project/
├── data/
│   ├── Fake.csv                     # Fake news articles
│   └── True.csv                     # Real news articles
├── Transformer_ML_Notebook.ipynb    # Main analysis notebook
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Important Notes

### Model Considerations
- **High Accuracy Caveat:** The 99.97% accuracy suggests the dataset may have very clear distinguishing features between real and fake news. This is a valuable finding - it shows that fake news in this dataset has identifiable patterns.
- **Generalization:** Performance on this specific dataset may not generalize to all types of misinformation
- **Dataset Bias:** Model trained on 2016-2017 era news; language patterns may differ in current articles

### Limitations
- Trained on English-language news only
- May not generalize well to satire or opinion pieces
- Does not fact-check specific claims, only classifies article credibility based on writing patterns

## Future Improvements

- Test on more recent and diverse news datasets
- Implement cross-validation for more robust evaluation
- Experiment with other transformer models (RoBERTa, DistilBERT, GPT)
- Add explainability features (attention visualization, important phrase extraction)
- Deploy as web API for real-time classification
- Test on multilingual datasets
- Fine-tune hyperparameters for even better performance
- Implement active learning for continuous improvement

## What I Learned

- **Transformer Architecture:** Practical experience with BERT and Hugging Face transformers library
- **NLP Pipeline:** Complete workflow from data preprocessing to model evaluation
- **Transfer Learning:** How to fine-tune pre-trained models for specific tasks
- **Model Comparison:** Importance of establishing baselines and comparing approaches
- **Performance Analysis:** Using confusion matrices and multiple metrics for thorough evaluation
- **Real-world Applications:** Understanding the social impact of AI in combating misinformation

## References

- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

## License

This project was created as the final project for EGR422 at California Baptist University.

## Author

**Collin Kerker**  
Computer Science Student @ California Baptist University  
Concentration: Machine Learning & AI

[GitHub](https://github.com/BetterButterBoy) | [LinkedIn](https://www.linkedin.com/in/collin-kerker/) | [Email](mailto:kerkercollin@gmail.com)

---

*This project demonstrates advanced NLP skills using transformer models for real-world misinformation detection.*
