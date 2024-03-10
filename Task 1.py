import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sms_data = pd.read_csv('sms_spam.csv')

#first few rows

print("First few rows of the dataset:")
print(sms_data.head())

# missing values

print("Missing values in the dataset:")
print(sms_data.isnull().sum())

#showing graphs

plt.figure(figsize=(8, 5))
sns.countplot(x='type', data=sms_data)
plt.title('Distribution of Spam and Non-Spam Messages')
plt.show()

#converting 'type' column into numerical values

sms_data['message_length'] = sms_data['text'].apply(len)
plt.figure(figsize=(12, 6))
sns.histplot(data=sms_data, x='message_length', hue='type', bins=50, kde=True)
plt.title('Distribution of Message Lengths for Spam and Non-Spam Messages')
plt.show()

#trained data set excecution

X_train, X_test, y_train, y_test = train_test_split(sms_data['text'], sms_data['type'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
classifier = MultinomialNB()

#checking accuracy
classifier.fit(X_train_vectorized, y_train)
y_pred = classifier.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))