# Sentiment-Analysis-for-Product-Review

Overview
This project performs sentiment analysis on a dataset of Amazon product reviews to classify them as positive, neutral, or negative. The goal is to build a robust model that can automatically determine the sentiment of customer feedback, which can be valuable for product improvement, customer service, and market research.

Dataset
The sentiment analysis is performed on a combined dataset sourced from multiple Amazon product review CSV files:

1429_1.csv
Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv
Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
The datasets contain various fields, but for this analysis, primarily reviews.text and reviews.rating columns are used.

Methodology
The project follows a standard Natural Language Processing (NLP) pipeline:

Data Loading and Merging: Multiple CSV files are loaded into Pandas DataFrames and then concatenated into a single DataFrame.
Data Cleaning: Missing values in reviews.text and reviews.rating are removed.
Sentiment Labeling: A custom function label_sentiment is applied to reviews.rating to create a sentiment column:
Ratings >= 4 are labeled as "positive".
Ratings == 3 are labeled as "neutral".
Ratings < 3 are labeled as "negative".
Text Preprocessing: The clean_text function performs the following steps:
Converts text to lowercase.
Removes non-alphabetic characters.
Tokenizes the text using spaCy.
Performs lemmatization.
Removes English stop words.
Filters out tokens with length less than or equal to 2.
Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned text reviews into numerical features. A TfidfVectorizer is configured with max_features=20000 and ngram_range=(1,2).
Model Training: A LinearSVC (Linear Support Vector Classifier) model from sklearn.svm is trained on the TF-IDF features.
Evaluation: The model's performance is evaluated using accuracy, F1-score, and a classification report.
Error Analysis: Misclassified reviews are identified and displayed to understand model weaknesses.
Installation
To run this notebook, you'll need the following Python libraries. You can install them using pip:

pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn
python -m spacy download en_core_web_sm
Also, ensure you have downloaded NLTK stopwords and punkt tokenizer:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
Usage
Clone the repository (if applicable).
Place the dataset files (1429_1.csv, Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv, Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv) in the specified path (e.g., /content/drive/MyDrive/ if running in Google Colab with Google Drive mounted).
Run the Jupyter Notebook cells sequentially.
Key Steps in the Notebook:
Data Loading: Cells loading df1, df2, df3.
Preprocessing: Cells defining label_sentiment, clean_text, and applying them to the DataFrame.
Model Training: Cells for TfidfVectorizer, train_test_split, and LinearSVC fitting.
Evaluation: Cells calculating accuracy_score, f1_score, classification_report, and plotting the confusion matrix.
Error Analysis: Cells to identify and display misclassified reviews.
Results
After training and evaluation, the model achieved the following performance metrics:

Accuracy: 0.9548
F1 Score (weighted): 0.9484
Classification Report:
              precision    recall  f1-score   support

    negative       0.84      0.64      0.73       502
     neutral       0.76      0.37      0.50       580
    positive       0.96      0.99      0.98     12510

    accuracy                           0.95     13592
   macro avg       0.86      0.67      0.73     13592
weighted avg       0.95      0.95      0.95     13592
Confusion Matrix:
(Insert Confusion Matrix Plot Here, e.g., by taking a screenshot or generating it dynamically if possible)

Observations from Misclassified Reviews:
(You can observe and add insights from misclassified.head(10) output here)

Many 'neutral' reviews were misclassified, often due to subtle phrasing that the model struggled to interpret as strictly neutral. For example, reviews mentioning minor inconveniences but overall satisfaction might be labeled positive by the user but predicted neutral or negative by the model, or vice-versa.
Some reviews labeled 'negative' in the original data might have positive words but an underlying negative context, which text preprocessing might not fully capture.
Future Improvements
Advanced Preprocessing: Experiment with more sophisticated text preprocessing techniques, including negation handling or sentiment-specific tokenization.
Different Models: Explore other machine learning models (e.g., Logistic Regression, Naive Bayes, Random Forest) or deep learning models (e.g., LSTMs, BERT-based models) for potentially better performance.
Hyperparameter Tuning: Optimize the hyperparameters of the TfidfVectorizer and LinearSVC for improved accuracy and F1-score.
Balanced Dataset: Address the class imbalance (predominantly positive reviews) using techniques like oversampling, undersampling, or synthetic data generation (SMOTE) to improve performance on minority classes (neutral, negative).
Error Analysis Enhancements: Perform a more in-depth qualitative analysis of misclassified reviews to identify common patterns and areas for improvement.
