import chazutsu
import re
import nltk
from nltk import sent_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')

imdb_review_data = chazutsu.datasets.IMDB().download()
train  = imdb_review_data.train_data()['review']
# ratings = imdb_review_data.train_data()['rating']
# polarity = imdb_review_data.train_data()['polarity']

text_data = []
stop_words = nltk.corpus.stopwords.words('english')
symbol = [ ':', ';', '.', ',', '-', '!', '?', "'s"]
stop_words = stop_words + symbol

for review in tqdm(train):
    sentences = sent_tokenize(review)
    for sent in sentences:
        sent = sent.lower().replace("\n", "").replace("\'", " ").replace("\"", " ")
        for sy in symbol: sent = sent.replace(sy, " ")
        sent = re.sub(r"[^a-z '\-]", "", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        token = sent.split(" ")
        token = [w.lower() for w in token if w.lower() not in stop_words]
    text_data.append([t for t in token if len(t) > 0])
    
# Convert text to a single string
text_data_joined = [" ".join(text) for text in text_data]
df = imdb_review_data.train_data()
df
# ----------------Q1-Q4---------------------#
# Add text_data, text_data_joined to df in case
df['text_data'] = text_data
df['text_data_joined'] = text_data_joined
df_sample = df.sample(n=6000, random_state=42).reset_index()
del df
del text_data
del text_data_joined
del train

# One-Hot Encoding
one_hot_vectorizer = CountVectorizer(binary=True)
one_hot_representation = one_hot_vectorizer.fit_transform(
    np.array(df_sample['text_data_joined'])).toarray()
df_sample['one_hot_representation'] = list(one_hot_representation)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_representation = tfidf_vectorizer.fit_transform(
    np.array(df_sample['text_data_joined'])).toarray()
df_sample['tfidf_representation'] = list(tfidf_representation)

model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=50000)

# Compute function for average Word2Vec vector
def average_word2vec(tokens_list, model, num_features=300):
    words = [word for word in tokens_list if word in model.key_to_index]
    if not words:
        return np.zeros(num_features)
    word_vectors = [model[word] for word in words]
    return np.mean(word_vectors, axis=0)

# Compute average Word2Vec vector for each text
average_vectors = [average_word2vec(review, model, 300) for review in np.array(
    df_sample['text_data'])]
df_sample['average_vectors'] = average_vectors

# Function to compute TF-IDF weighted average vector
def tfidf_weighted_word2vec(tokens_list, model, tfidf_row, feature_names, 
                            num_features=300):
    weighted_word_vectors = []
    for word in tokens_list:
        if word in model.key_to_index and word in feature_names:
            # Get TF-IDF value
            index = np.where(feature_names == word)[0][0]
            tfidf_value = tfidf_row[index]
            # Multiply Word2Vec vector by TF-IDF value and add to list for weighted average
            weighted_word_vectors.append(model[word] * tfidf_value)

    if not weighted_word_vectors:
        return np.zeros(num_features)

    # Compute weighted average Word2Vec vector
    return np.mean(weighted_word_vectors, axis=0)

# Compute TF-IDF weighted average Word2Vec vector for each text
tfidf_weighted_vectors = [tfidf_weighted_word2vec(review, model, np.array(
    df_sample['tfidf_representation'])[i], np.array(tfidf_vectorizer.get_feature_names_out()), 300) 
                          for i, review in enumerate(np.array(df_sample['text_data']))]
df_sample['tfidf_weighted_vectors'] = tfidf_weighted_vectors

#--------------------Q5, 6, 7--------------------------------------------#
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Calculate the sentiment score for each text
sentiment_scores = []
for text in tqdm(np.array(df_sample['text_data_joined'])):
    scores = sia.polarity_scores(text)
    sentiment_scores.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

X_sentiment_score = np.array(sentiment_scores)
y_rating = np.array(df_sample['rating'])
y_polarity = np.array(df_sample['polarity'])

# split data
X_one_hot = np.array(df_sample['one_hot_representation'].apply(
    lambda x: np.array(x)).tolist()).astype(float)
X_tfidf = np.array(df_sample['tfidf_representation'].apply(
    lambda x: np.array(x)).tolist()).astype(float)
X_average = np.array(df_sample['average_vectors'].apply(
    lambda x: np.array(x)).tolist()).astype(float)
X_tfidf_weighted = np.array(df_sample['tfidf_weighted_vectors'].apply(
    lambda x: np.array(x)).tolist()).astype(float)

(X_sentiment_score_train, X_sentiment_score_test, y_rating_train, y_rating_test, 
y_polarity_train, y_polarity_test) = train_test_split(
    X_sentiment_score, y_rating, y_polarity, test_size=0.2, random_state=42)

#-------- For rating（Apply multinomial logistic model）-------#
# We do not care now about the data in the "y" part, since they are acquired separately, but only the split data in the "X" part is acquired.
X_train_one_hot, X_test_one_hot, _, _ = train_test_split(
    X_one_hot, y_rating, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
    X_tfidf, y_rating, test_size=0.2, random_state=42)
X_train_average, X_test_average, _, _ = train_test_split(
    X_average, y_rating, test_size=0.2, random_state=42)
X_train_tfidf_weighted, X_test_tfidf_weighted, _, _ = train_test_split(
    X_tfidf_weighted, y_rating, test_size=0.2, random_state=42)

def train_and_predict_rating(X_train, X_test, y_train):
    model = LogisticRegressionCV(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

y_pred_one_hot = train_and_predict_rating(X_train_one_hot, X_test_one_hot, y_rating_train)
y_pred_tfidf = train_and_predict_rating(X_train_tfidf, X_test_tfidf, y_rating_train)
y_pred_average = train_and_predict_rating(X_train_average, X_test_average, y_rating_train)
y_pred_tfidf_weighted = train_and_predict_rating(
    X_train_tfidf_weighted, X_test_tfidf_weighted, y_rating_train)
y_pred_sentiment_score = train_and_predict_rating(
    X_sentiment_score_train, X_sentiment_score_test, y_rating_train)

# Model evaluation
accuracy_one_hot = accuracy_score(y_rating_test, y_pred_one_hot)
accuracy_tfidf = accuracy_score(y_rating_test, y_pred_tfidf)
accuracy_average = accuracy_score(y_rating_test, y_pred_average)
accuracy_tfidf_weighted = accuracy_score(y_rating_test, y_pred_tfidf_weighted)
accuracy_sentiment_scores = accuracy_score(y_rating_test, y_pred_sentiment_score)

print("Accuracy for One-Hot Encoding:", accuracy_one_hot)
print("Accuracy for TF-IDF:", accuracy_tfidf)
print("Accuracy for Average Word2Vec:", accuracy_average)
print("Accuracy for TF-IDF Weighted Word2Vec:", accuracy_tfidf_weighted)
print("Accuracy for sentiment score:", accuracy_sentiment_scores)

#--------- For polarity（Apply logistic regression）-------#
# Data splitting is not necessary here, since we will use the above data for X.
def train_and_predict_polarity(X_train, X_test, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred_proba

# Model training and prediction for each feature
y_pred_proba_one_hot = train_and_predict_polarity(
    X_train_one_hot, X_test_one_hot, y_polarity_train)
y_pred_proba_tfidf = train_and_predict_polarity(
    X_train_tfidf, X_test_tfidf, y_polarity_train)
y_pred_proba_average = train_and_predict_polarity(
    X_train_average, X_test_average, y_polarity_train)
y_pred_proba_tfidf_weighted = train_and_predict_polarity(
    X_train_tfidf_weighted, X_test_tfidf_weighted, y_polarity_train)
y_pred_proba_sentiment_score = train_and_predict_polarity(
    X_sentiment_score_train, X_sentiment_score_test, y_polarity_train)

# Threshold
threshold = 0.5

# Binarize each prediction
y_pred_proba_one_hot_binary = (y_pred_proba_one_hot > threshold).astype(int)
y_pred_proba_tfidf_binary = (y_pred_proba_tfidf > threshold).astype(int)
y_pred_proba_average_binary = (y_pred_proba_average > threshold).astype(int)
y_pred_proba_tfidf_weighted_binary = (y_pred_proba_tfidf_weighted > threshold).astype(int)
y_pred_proba_sentiment_score_binary = (y_pred_proba_sentiment_score > threshold).astype(int)

# Model evaluation
accuracy_proba_one_hot = accuracy_score(y_polarity_test, y_pred_proba_one_hot_binary)
accuracy_proba_tfidf = accuracy_score(y_polarity_test, y_pred_proba_tfidf_binary)
accuracy_proba_average = accuracy_score(y_polarity_test, y_pred_proba_average_binary)
accuracy_proba_tfidf_weighted = accuracy_score(
    y_polarity_test, y_pred_proba_tfidf_weighted_binary)
accuracy_proba_sentiment_scores = accuracy_score(
    y_polarity_test, y_pred_proba_sentiment_score_binary)

print("Accuracy for One-Hot Encoding:", accuracy_proba_one_hot)
print("Accuracy for TF-IDF:", accuracy_proba_tfidf)
print("Accuracy for Average Word2Vec:", accuracy_proba_average)
print("Accuracy for TF-IDF Weighted Word2Vec:", accuracy_proba_tfidf_weighted)
print("Accuracy for sentiment score:", accuracy_proba_sentiment_scores)
#--------------------------------------------------------#
# Draw ROC curve
def plot_roc_curve(y_true, y_pred_proba, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.figure(figsize=(10, 6))
plot_roc_curve(y_polarity_test, y_pred_proba_one_hot, 'One-Hot')
plot_roc_curve(y_polarity_test, y_pred_proba_tfidf, 'TF-IDF')
plot_roc_curve(y_polarity_test, y_pred_proba_average, 'Average Word2Vec')
plot_roc_curve(y_polarity_test, y_pred_proba_tfidf_weighted, 'TF-IDF Weighted Word2Vec')
plot_roc_curve(y_polarity_test, y_pred_proba_sentiment_score, 'Sentiment Score')

plt.title('ROC Curve for Polarity Prediction')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('ROC Curve for Polarity Prediction')
plt.show()
