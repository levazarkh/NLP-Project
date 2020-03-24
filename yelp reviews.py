import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


# reading the data and setting it up as a dataframe
yelp = pd.read_csv('yelp.csv')

# checking info on yelp
print(yelp.head())
print(yelp.info())
print(yelp.describe())

# creating new column "text length"
yelp['text length'] = yelp['text'].apply(len)

# exploratory data analysis
# creating a grid of 5 histograms of text length based off of the star ratings
g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length')
plt.show()

# number of occurrences for each type of star rating
sns.countplot(x='stars', data=yelp, palette='rainbow')
plt.show()

stars = yelp.groupby('stars').mean()
print(stars)
print(stars.corr())

sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)
plt.show()

# NLP classification task
# creating dataframe that contains the columns only of 1 and 5 stars
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

X = yelp_class['text']
y = yelp_class['stars']

# creating CountVectorizer object
cv = CountVectorizer()
X = cv.fit_transform(X)

# spliting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# training a model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# predictions and evaluations
predictions = nb.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

# using text processing
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
