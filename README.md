# NLP-Project--Tamil-Sentiment-Analysis

# 🎭 Sentiment Analysis of Code-Mixed Tamil-English and Pure Tamil YouTube Comments with Sentiment-Based Summarization

This project presents a comprehensive Natural Language Processing (NLP) pipeline for performing **real-time sentiment analysis and summarization** of Tamil-English **code-mixed** and **pure Tamil** YouTube comments. It integrates **machine learning**, **transformer-based models**, **YouTube data scraping**, and a **Gradio-powered user interface** for interactive use.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Dataset Details](#dataset-details)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [How It Works](#how-it-works)
- [Outputs](#outputs)
- [Acknowledgements](#acknowledgements)

---

## 📖 Overview

With the rising popularity of social media in regional languages, especially Tamil, YouTube users often post comments in **code-mixed Tamil-English** or **pure Tamil**. Standard NLP models struggle with such linguistic diversity.

This project:
- Automatically scrapes YouTube comments in real-time.
- Detects language type (code-mixed or pure Tamil).
- Classifies sentiment using traditional ML and fine-tuned transformer models.
- Summarizes comments by sentiment category.
- Presents results through a Gradio web interface.

---

## ✨ Key Features

✅ **Dual Sentiment Classification**  
✅ **Language Detection (Tamil or Code-Mixed)**  
✅ **Transformer Models (BERT & IndicBERT)**  
✅ **YouTube Comment Scraping (YouTube API)**  
✅ **Text Summarization using BART**  
✅ **Evaluation Metrics (Accuracy, F1, ROUGE, BLEU)**  
✅ **CSV Export of Results**  
✅ **Gradio Interface for Real-Time Interaction**

---

## 🧱 Architecture

The project architecture is divided into two core pipelines: Model Training & Comparison and Real-Time YouTube Sentiment Analysis. The first pipeline covers data preparation, model training (Logistic Regression, Random Forest, BERT/IndicBERT), and evaluation to output labeled test datasets. The second pipeline begins with YouTube video input, followed by comment scraping, preprocessing, sentiment classification using BERT/IndicBERT, and BART-based summarization. Results, including sentiment summaries and video metadata, are presented via a Gradio interface with CSV export functionality.


### 🔹 Model Comparison Architecture

![image](https://github.com/user-attachments/assets/48562168-5fbc-4d2a-bd75-2cd01cdaf3f0)

### 🔹 Youtube Architecture

![image](https://github.com/user-attachments/assets/47609202-3abd-421e-9d2a-785e26657aeb)


The system follows a modular pipeline that begins with user input via a YouTube video link. Comments are scraped using the YouTube API and passed through a preprocessing stage involving tokenization, cleaning, and language detection. Depending on whether the comment is code-mixed or pure Tamil, it is classified using a BERT-based or IndicBERT-based model respectively. The output sentiments are displayed through a Gradio interface, and results—along with metadata and sentiment summaries—are exported as downloadable CSV files.


---

## 💻 Technologies Used

| Component               | Technology                                   |
|------------------------|----------------------------------------------|
| Web Scraping           | YouTube Data API                             |
| ML Models              | Logistic Regression, Random Forest           |
| Transformer Models     | BERT (Code-Mixed), IndicBERT (Pure Tamil)    |
| Summarization Model    | Facebook BART-large-CNN                      |
| Interface              | Gradio                                       |
| Text Processing        | NLTK, Scikit-learn, TensorFlow, HuggingFace  |
| Evaluation Metrics     | ROUGE, BLEU, Cosine Similarity               |
| Visualization          | Matplotlib, Seaborn                          |

---

## 📊 Dataset Details

The sentiment analysis in this project is powered by two key datasets: a **Pure Tamil dataset** and a **Code-Mixed Tamil-English dataset**, each divided into training and testing subsets. All datasets consist of a single feature — text — labeled into three sentiment classes: Negative, Neutral, and Positive.

The **Pure Tamil training set** contains 4,873 samples, with 1,783 negative, 1,607 neutral, and 1,483 positive comments. The **Pure Tamil test set** includes 541 samples distributed as 197 negative, 169 neutral, and 175 positive.

For code-mixed text, the **Code-Mixed Tamil-English training set** is larger, with 7,872 samples, comprising 2,632 negative, 2,625 neutral, and 2,615 positive comments. The **test set** for this category has 2,199 entries, with 742 samples each for negative and neutral sentiments, and 715 labeled as positive.

These datasets are sourced from two prominent repositories:

- 🔗 [Dravidian-CodeMix 2020 Dataset](https://dravidian-codemix.github.io/2020/datasets.html)
- 🔗 [MACD Dataset Meta from ShareChatAI](https://github.com/ShareChatAI/MACD/tree/main/dataset_meta)

These curated datasets provide rich linguistic diversity and class distribution necessary for building robust multilingual and code-mixed sentiment classification models, especially for low-resource languages like Tamil.

---

## 🔍 Model Details

| Model              | Dataset           | Accuracy   |
|-------------------|-------------------|------------|
| Logistic Regression | Code-Mixed        | ~78%       |
| Random Forest       | Code-Mixed        | ~99% (Overfitted) |
| BERT (Fine-Tuned)   | Code-Mixed        | 87%        |
| Logistic Regression | Pure Tamil        | 86%        |
| Random Forest       | Pure Tamil        | 86%        |
| IndicBERT (Fine-Tuned) | Pure Tamil    | 80%        |

---

## 📈 Performance Metrics

- **Accuracy** (Train & Validation)
- **Confusion Matrices**
- **Classification Reports (Precision, Recall, F1)**
- **Summarization Evaluation**:
  - ROUGE-1, ROUGE-L
  - BLEU (Cosine Similarity Approximation)

---

## 🧪 How It Works

1. **Scrape YouTube Comments**: 
   - Enter a YouTube URL in the Gradio app.
   - Comments are scraped via the YouTube API.

2. **Preprocess & Detect Language**:
   - Text is cleaned and checked for Tamil script.
   - Classify as pure Tamil or code-mixed.

3. **Run Dual Sentiment Analysis**:
   - Code-Mixed → BERT
   - Pure Tamil → IndicBERT
   - Alternatives: Logistic Regression, Random Forest for comparison

4. **Summarize Comments**:
   - Top 50 comments per class → summarized using BART

5. **Evaluate Summary Quality**:
   - Using ROUGE, BLEU metrics

6. **Visualize & Export**:
   - Metadata, summaries, and CSV download provided via Gradio.

---

## 📤 Outputs
🎬 Video Metadata (Title, Views, Channel, Duration)

📋 Sentiment Label for Each Comment

📝 Summaries for Positive, Negative, Neutral Comments

📁 Downloadable CSV (Sentiments + Comments)



---

## 🙏 Acknowledgements
HuggingFace Transformers
AI4Bharat for IndicBERT
Facebook for BART
Google YouTube API





