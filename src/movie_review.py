

#import libraries
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from google.colab import files
nltk.download('vader_lexicon')
data=pd.read_excel('/content/movie_reviews.xlsx')

def durable(d):
  if isinstance(d, str):
    hr=min=sec=0
    h=re.search(r'(\d+)H',d)
    m=re.search(r'(\d+)M',d)
    s=re.search(r'(\d+)S',d)
    if h:
      hr=int(h.group(1))
    if m:
      min=int(m.group(1))
    if s:
      sec=int(s.group(1))
    return hr*60+min+sec/60
  elif isinstance(d, (int, float)):
    return d
  else:
    return np.nan

# Data cleaning, handling, preprocessing
data.dropna()
data.drop_duplicates(subset=["IMDB Ids"],keep='first',inplace=True)
data.drop(columns=["Serial No","URLs","Title"])
data['Age of content'] = data['Age of content'].fillna(data['Age of content'].median())

#sentiment scoring by vader
sia = SentimentIntensityAnalyzer()
data['vader_score'] = data['Review'].fillna('').astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
data['vader_score'] = pd.to_numeric(data['vader_score'], errors='coerce')
data['label'] = data['vader_score'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
##X = data[metadata_features + ['review']]
##y = data['label']

# genre list
data = data.loc[:, ~data.columns.isin(['[', ']'])]
data = data.loc[:, ~data.columns.duplicated()]
data['Genre'] = data['Genre'].astype(str).str.replace(r'[\[\]\']', '', regex=True)
data['Genre'] = data['Genre'].str.lower().str.strip()
genre_mapping = {
    'musical': 'music',
    'sci-fi': 'science fiction',
    'romcom': 'romance',
    'thrillers': 'thriller',
    'adventures': 'adventure'
}
data['Genre'] = data['Genre'].replace(genre_mapping)
mlb=MultiLabelBinarizer()
data['Genre_list'] = data['Genre'].fillna('').astype(str).apply(lambda x: [g.strip() for g in re.split('[,;/"]', str(x)) if g.strip()])
data_exploded_genre = data.explode('Genre_list')
genre_encoded=pd.DataFrame(mlb.fit_transform(data['Genre_list']),columns=mlb.classes_,index=data.index)
data=pd.concat([data,genre_encoded],axis=1)

# data preprocessing: new labels
data['plot_keywords_list'] = data['Plot Keywords'].fillna('').apply(lambda x: [w.strip().lower() for w in x.split(',') if w.strip()])
# cast
data['Cast_list'] = data['Cast'].fillna('').astype(str).apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
data['cast_count'] = data['Cast_list'].apply(len)
# duration
data['duration_minutes']= data['Duration'].apply(durable)
print(data.columns.tolist)

# data visualisation
# Aggregate vader_score by Year_Released and plot
mean_score_by_year = data.groupby('Year_Released')['vader_score'].mean()
plt.figure(figsize=(12, 10))
plt.subplot(2,2,1)
plt.plot(mean_score_by_year.index[::10], mean_score_by_year.values[::10])
plt.title('Mean Vader Score Over Years')
plt.xlabel('Year Released')
plt.ylabel('Mean Vader Score')

# Aggregate vader_score by Age of content and plot
mean_score_by_age = data.groupby('Age of content')['vader_score'].mean()
plt.subplot(2,2,2)
plt.bar(pd.to_numeric(mean_score_by_age.index, errors='coerce'), mean_score_by_age.values)
plt.title('Mean Vader Score by Age of Content')
plt.xlabel('Age of Content')
plt.ylabel('Mean Vader Score')

# Aggregate vader_score by Genre and plot
plt.subplot(2,2,3)
mean_score_by_genre = data_exploded_genre.groupby('Genre_list')['vader_score'].mean()
mean_score_by_genre = mean_score_by_genre.sort_values(ascending=False)
plt.bar(mean_score_by_genre.index, mean_score_by_genre.values)
plt.title('Mean VADER Score by Genre')
plt.xlabel('Genre')
plt.ylabel('Mean VADER Score')
plt.xticks(rotation=90)

# Aggregate vader_score by Duration and plot (assuming Duration is numerical)
# If 'Duration' has many unique values, a scatter plot or binning might be more appropriate.
# For now, aggregating and plotting mean score for each duration.
mean_score_by_duration = data.groupby('duration_minutes')['vader_score'].mean()
mean_score_by_duration = mean_score_by_duration.sort_index() # Sort by duration
plt.subplot(2,2,4)
plt.barh(mean_score_by_duration.index[::10], mean_score_by_duration.values[::10]) # Scatter plot for duration
plt.title('Mean Vader Score by Duration')
plt.xlabel('Duration')
plt.ylabel('Mean Vader Score')
plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

plt.savefig('genre_sentiment.png',bbox_inches='tight')
plt.show()
files.download('genre_sentiment.png')

# metadata
metadata_features = ['cast_count', 'duration_minutes', 'Age of content']+list(genre_encoded.columns)
X=data[metadata_features+['Review']]
y=data['label']
tfidf=TfidfVectorizer(max_features=2000,stop_words='english')
X_text=tfidf.fit_transform(data['Review'].astype(str))
X_meta= data[metadata_features].fillna(0)
X_combo=hstack([X_text,X_meta])
X_train, X_test, y_train, y_test=train_test_split(X_combo,y,test_size=0.3,random_state=42,stratify=y)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Classification Report: ", classification_report(y_test,y_pred))
feature_names = np.array(tfidf.get_feature_names_out().tolist() + metadata_features)
importance = model.coef_[0]
top_indices = importance.argsort()[-20:][::-1]
print("Top Positive Indicators:\n", feature_names[top_indices])
print("total reviews=",data['Genre_list'].nunique)

positive = (data['vader_score'] > 0.05).sum()
negative = (data['vader_score'] < -0.05).sum()
neutral  = ((data['vader_score'] >= -0.05) & (data['vader_score'] <= 0.05)).sum()

total = positive + negative + neutral
pos_percent = round((positive / total) * 100, 2)
neg_percent = round((negative / total) * 100, 2)
neu_percent = round((neutral  / total) * 100, 2)
mean_vader = round(data['vader_score'].mean(), 3)

print("\nSentiment Distribution:")
print(f"Positive: {pos_percent}% | Negative: {neg_percent}% | Neutral: {neu_percent}%")
print("Mean VADER Score:", mean_vader)

acc = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Performance:")
print(f"Accuracy: {acc:.3f}")

avg_duration = data['duration_minutes'].mean()
most_common_age = data['Age of content'].mode()[0]
print("Average Duration (min):", avg_duration)
print("Most Common Age Rating:", most_common_age)