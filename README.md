# Project Overview

The goal of this project is the detection of fake news by using machine learning to classify news articles as either fake or real. This is important for many reasons, mainly because of the huge spread of false information through online news articles. This spread occurs because of the unfiltered and generally “free to post whatever you want” nature of the internet, meaning that fake information can be spread at rapid and alarming rates, calling for the need for something that can detect whether news is actually real or not.

For this project, I used a fine-tuned BERT transformer model trained on a dataset of about 40,000 real and fake labeled news articles from Kaggle. I chose BERT because of how well it understands contextual meaning in text. This makes it perfect for reading often complex news articles with very deep context.

The results turned out very good. A baseline model using TF-IDF and Logistic Regression was created for comparison against the main model. After training the main model, BERT achieved very good performance even though there was very little hyperparameter tuning during training. It achieved a very solid score for detecting the news, having about 99% accuracy in identifying real and fake news alike.

# Installation / Setup Instructions

## Dependecies Intall (use this line to install all of the required dependecies): 
- pip install scikit-learn pandas numpy matplotlib seaborn transformers datasets accelerate torch
- Additional downloads being python (recommended version 3.11) and Jupyter Notebook

# Results Summary

## General Metrics Results
Accuracy: ~99%
Precision / Recall / F1: 0.999
Eval Loss: 0.0026

These results are very good considering the model was trained with minimal hyperperameter tuning and was run off of the CPU. BERT showed good performance in identifying the real vs fake news articles.

## Confusion Matrix Results
- Fake news predicted as fake: 3495
- Fake news predicted as real: 0
- Real news predicted as real: 4233
- Real news predicted as fake: 2

## Interpretation
Overall, BERT performed exceptionally well at idetifying real vs fake news, with the accuracy for the real news being ever so slightly worse than the fake news. This is not bad at all through because it being a little skeptical is good and the real purpose of this model was to detect fake news anyways