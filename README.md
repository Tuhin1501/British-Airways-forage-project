# British-Airways-forage-project
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import re
from sklearn.utils import shuffle
import string
import nltk
nltk.download('punkt')
import requests
from tqdm.notebook import tqdm
import plotly as pt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn import naive_bayes
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,  RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
# This function displays the splits of the tree
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)
from sklearn.ensemble import RandomForestClassifier
# This module lets us save our models once we fit them.
import pickle
from sklearn.model_selection import PredefinedSplit
# This is the classifier
from xgboost import XGBClassifier
# This is the function that helps plot feature importance 
from xgboost import plot_importance

nltk.download('averaged_perceptron_tagger')
base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100
reviews = []
# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")

df = pd.DataFrame()
df["reviews"] = reviews

df.shape

#I took off the extra emojis from texts and cleaned it on excel then imported it onto python
data = pd.read_csv(r"C:\Users\t1u5h\OneDrive\Documents/BA_reviews.csv")

data = data.drop(columns="Unnamed: 0")
df["Trip_Verification"].value_counts()
example = df["Customer_Reviews"][100]
example
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
tagged[:10]
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
 nltk.download('vader_lexicon')
 sia = SentimentIntensityAnalyzer()
 sia.polarity_scores('I am okay')
 sia.polarity_scores(example)


resdata = pd.DataFrame(res)
res={}
for i, row in tqdm(data.iterrows(), total=len(data)):
    review = row["Customer_Reviews"]
    verification = row["Trip_Verification"]
    res[verification] = sia.polarity_scores(review)
res

results = pd.DataFrame(res).T

 results = results.reset_index()

 fulldata = pd.merge(results, data, on = "index", how = "left")
 fulldata["Analysis"] = pd.cut(x= fulldata["compound"], bins=[-1,0,0.5,1], labels=["Negative","Neutral","Positive"])
 fulldata["compound"] = round(fulldata["compound"],2)
 fulldata["Analysis"].value_counts()
 import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(15,7))

plt.subplot(1,3,2)
plt.title("Reviews Analysis")
plt.pie(fulldata["Analysis"].value_counts(), labels= fulldata["Analysis"].value_counts().index,
        explode = (0, 0, 0.25), autopct='%1.1f%%', shadow=False)

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(fulldata.Customer_Reviews)


# Applying ml algorithms to predict customer booking

data = pd.read_csv(r"C:\Users\t1u5h\Downloads\customer_booking.csv", encoding="ISO-8859-1")
data["sales_channel"].unique()
data["trip_type"].value_counts(normalize = True)
data["Route"].unique()
data["sales_channel"].value_counts(normalize = True)
data["flight_day"].value_counts(normalize = True)
Y = data[["booking_complete"]]
X = data.drop(columns=["booking_complete"])
X["sales_channel"] = X["sales_channel"].map({'Internet':0,'Mobile':1})
X["trip_type"] = X["trip_type"].map({'RoundTrip':2, 'CircleTrip':1, 'OneWay':0})
X["flight_day"].unique()
X["flight_day"] = X["flight_day"].map({'Sat':6, 'Wed':3, 'Thu':4, 'Mon':1, 'Sun':7, 'Tue':2, 'Fri':5})
route_1 = (X.groupby('Route').size()) / len(X)
X["Route"] = X['Route'].apply(lambda x : route_1[x])
booking_origin_1 = (X.groupby('booking_origin').size()) / len(X)
X["booking_origin"] = X['booking_origin'].apply(lambda x : booking_origin_1[x])
scaler = MinMaxScaler()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=0)
x_tr,x_val,y_tr,y_val = train_test_split(x_train,y_train,test_size=0.25,stratify=y_train, random_state=0)
DT = DecisionTreeClassifier(random_state=0)
DT.fit(x_tr,y_tr)
y_pred_val=DT.predict(x_val)
recall_score(y_val,y_pred_val)
accuracy_score(y_val,y_pred_val)
precision_score(y_val,y_pred_val)
f1_score(y_val,y_pred_val)
DT.fit(x_train,y_train)
y_pred=DT.predict(x_test)
recall_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
f1_score(y_test,y_pred)
RF = RandomForestClassifier(random_state=0)
RF.fit(x_tr,y_tr)
y_pred_val=RF.predict(x_val)
recall_score(y_val,y_pred_val)
accuracy_score(y_val,y_pred_val)
precision_score(y_val,y_pred_val)
f1_score(y_val,y_pred_val)
xgb = XGBClassifier(objective='binary:logistic', random_state=0)
xgb.fit(x_tr,y_tr)
y_pred_val=xgb.predict(x_val)
recall_score(y_val,y_pred_val)
accuracy_score(y_val,y_pred_val)
precision_score(y_val,y_pred_val)
f1_score(y_val,y_pred_val)
y_pred=xgb.predict(x_test)
recall_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
f1_score(y_test,y_pred)
y_pred=RF.predict(x_test)
recall_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
f1_score(y_test,y_pred)
RF.feature_importances_
global_importances = pd.Series(RF.feature_importances_, index=x_train.columns)
global_importances.sort_values(ascending=True, inplace=True)
global_importances.plot.barh(color='green')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Global Feature Importance - Built-in Method")


