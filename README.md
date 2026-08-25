# 📰 Fake News Detection

A **Machine Learning-based Fake News Detection System** that classifies news articles as **Fake or Real** using Natural Language Processing (NLP) and TF-IDF feature extraction.

###  Technologies

* Python
* Pandas & NumPy
* Scikit-learn
* TF-IDF
* Matplotlib & Seaborn
* Joblib
* Tkinter

###  ML Models

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest

###  Features

* Fake/Real news classification
* NLP text preprocessing
* TF-IDF feature extraction
* Multiple model comparison
* Accuracy, Precision, Recall & F1-score evaluation
* Confusion matrix and model comparison graphs
* Automatic selection of the best-performing model
* Tkinter-based GUI

###  Best Model

Among the tested models, **SVM achieved the highest accuracy of approximately 99.39%** on the test dataset.

###  Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python src/fake_news_gui.py
```

###  Project Structure

```text
Fake_News_Detection/
├── dataset/
├── models/
├── outputs/
├── src/
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── predict.py
│   └── fake_news_gui.py
└── README.md
```

**Author:** Meghana R. Naragund
B.E. Artificial Intelligence & Machine Learning
