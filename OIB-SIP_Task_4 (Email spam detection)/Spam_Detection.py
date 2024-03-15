# %% [markdown]
# # Import Necessory Library

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import r2_score
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# %% [markdown]
# # Load dataset

# %%
data = pd.read_csv("spam.csv",encoding="latin1")
data.head()

# %% [markdown]
# # EDA

# %%
data.isnull().sum()

# %%
data.dropna(axis=1,inplace=True)
data.head()

# %%
data.info()

# %%
data.columns=['Label','Mail']

# %%
data.duplicated().sum()

# %%
data.drop_duplicates(inplace=True)

# %%
data.shape

# %%
data.info()

# %%
plt.pie(data['Label'].value_counts(),labels=['ham','spam'],autopct='%.2f')
plt.show()


# %% [markdown]
# Label Encoding into target column

# %%
le = LabelEncoder()

# %%
data['Label']=le.fit_transform(data['Label'])

# %%
data.head()

# %% [markdown]
# # Text processing

# %%
# Converting the text in lowercase

def process_text(text):
  text=text.lower()

#tokenization
  text=nltk.word_tokenize(text)

# removing stopwords an punctuation
  x=[]
  for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
        x.append(i)

  text=x[:]
  x.clear( )

# stemming

  ps=PorterStemmer()
  for i in text:
    x.append(ps.stem(i))


  return " ".join(x)

# %%
data['Transformed_mail']=data['Mail'].apply(process_text)


# %%
data.head()

# %%
tfidf = TfidfVectorizer(max_features = 3000)

# %%
X = tfidf.fit_transform(data['Transformed_mail']).toarray()

# %%
X

# %%
y = data['Label'].values


# %% [markdown]
# # Split data into train and test

# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=47)

# %% [markdown]
# # Model build

# %%
log_reg = LogisticRegression()

# %%
log_reg.fit(X_train,y_train)

# %%
train_pred = log_reg.predict(X_train)

# %%
test_pred = log_reg.predict(X_test)

# %%
print(f''' The train accuracy : {r2_score(y_train,train_pred)} 
The test accuracy : {r2_score(y_test , test_pred)}''')

# %%
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
train_prediction = bnb.predict(X_train)
test_prediction = bnb.predict(X_test)
print(f'''The train accuracy : {r2_score(y_train,train_prediction)}
The test accuracy : {r2_score(y_test , test_prediction)}''')

# %%
import joblib

# Assuming your trained model is named 'model'
joblib.dump(log_reg, 'log_model.pkl')

# %%
# Assuming your TF-IDF vectorizer is named 'vectorizer'
joblib.dump(tfidf, 'vectorizer.pkl')


# %%



