#!/usr/bin/env python
# coding: utf-8

# # [My Github](https://github.com/SanjeevKurJha/Natural-Language-Processing)

# In[1]:


#Importing the library 
import numpy as np
import pandas as  pd
import nltk
import time
import collections
nltk.download('punkt')
import re
import matplotlib.pyplot as plt
nltk.download('inaugural')
from nltk.tokenize import word_tokenize,sent_tokenize 
import urllib.request
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('averaged_perceptron_tagger')
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer


# # Build word frequency model based on all Tweets

# # Importing the data

# In[2]:


def read_data(filename):
    data = pd.read_csv(filename, sep=',')
    return data


# In[3]:


data_set = read_data('TweetSent.csv')


# # Exploratory data analysis

# In[4]:


data_set.head()


# In[5]:


data_set.tail()


# In[6]:


print(data_set.describe())


# In[7]:


print(data_set.info())


# In[8]:


data_Tweets=data_set['Tweet'].astype(str)


# # Data Preprocessing

# In[9]:


def clean_text(text):
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"you'r", "you are", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r" \'m", " am", text)  
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"they'r", "they are", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"this's", "this is", text)
    text = re.sub(r"'what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r'"' , "", text)
    text = re.sub(r"'" , "", text)
    text = re.sub(r"[0-9]+" , "", text)
    text = re.sub(r"<b>" , "", text)
    text = re.sub(r"<i>" , "", text)
    text = re.sub(r"<" , "", text)
    text = re.sub(r">" , "", text)  
    text = re.sub(r"[~`!@#$%^&*_=():;/?_+|,.-]","",text)
    return text


# In[10]:


stop_words = set(stopwords.words('english'))
def text_prepare(text):
    word_text=""
    text = text.lower()
    for word in text.split(" "): 
        if word.startswith("https://") or word.startswith("http://"):
            word=" " 
        elif word.startswith("@"):
            word=" " 
        else:
            if word not in stop_words:
                word=clean_text(word)
                word_text=word_text+" "+word
    return word_text


# In[11]:


cleaned_tweets = ""
cleaned_text = ""
cleaned_sent = ""
for data_Tweet in data_Tweets:
    cleaned_text=text_prepare(data_Tweet)
    cleaned_sent=cleaned_sent + cleaned_text
    cleaned_tweets=cleaned_tweets + cleaned_text +"."
    


# In[12]:


cleaned_sent


# In[13]:


cleaned_tweets


# # Tokenize the data

# In[14]:


word_tokens = nltk.word_tokenize(cleaned_sent)
print(word_tokens)


# # Count each word in sentances 

# In[15]:


words_counts = {}
for word in word_tokens:
    if word in words_counts:
        words_counts[word] = words_counts[word] + 1
    else:
        words_counts[word] = 1 
print(words_counts)


# # Find most common 20 words

# In[16]:


most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:20]
print(most_common_words)


# # Frequency distributions

# In[17]:


import operator
get_ipython().run_line_magic('matplotlib', 'inline')
Freq_dist_nltk=nltk.FreqDist(word_tokens)
print(Freq_dist_nltk)
sorted_d = sorted(Freq_dist_nltk.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_d[:25])
Freq_dist_nltk.plot(25, cumulative=False)


# # Building the wordcloud
# 

# In[18]:


from wordcloud import WordCloud
wordcloud = WordCloud().generate(cleaned_sent)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # Build word frequency model for  Tweets group by category

# In[85]:


Clean_sent_arr=[]
for cleaned_tweet in cleaned_tweets.split("."):
    Clean_sent_arr.append(cleaned_tweet)


# In[20]:


df = pd.DataFrame(Clean_sent_arr,columns =['Tweet']) 


# In[21]:


df_col_merge =pd.concat([df, data_set['Category']], axis=1)


# In[22]:


dataset = df_col_merge.dropna(how='any',axis=0)


# # Exploratory Data Analysis 

# In[23]:


dataset.head()


# In[24]:


dataset.tail()


# In[25]:


dataset.describe()


# In[27]:


dataset.info()


# # Filter Negative dataset and build word frequency 

# In[28]:


dataset_Negative=dataset[dataset['Category'] == "negative"]


# In[29]:


dataset_Negative=dataset_Negative['Tweet']


# In[30]:


type(dataset_Negative)


# In[31]:


cleaned_text = ""
cleaned_sent = ""
for dataset_Tweet in dataset_Negative:
    cleaned_text=text_prepare(dataset_Tweet)
    cleaned_sent=cleaned_sent + cleaned_text
    


# # Tokenize the data

# In[32]:


word_tokens = nltk.word_tokenize(cleaned_sent)


# In[33]:


words_counts = {}
for word in word_tokens:
    if word in words_counts:
        words_counts[word] = words_counts[word] + 1
    else:
        words_counts[word] = 1 


# In[34]:


most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:20]


# In[35]:


Freq_dist_nltk=nltk.FreqDist(word_tokens)
print(Freq_dist_nltk)
sorted_d = sorted(Freq_dist_nltk.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_d[:25])
Freq_dist_nltk.plot(25, cumulative=False)


# In[36]:


wordcloud = WordCloud().generate(cleaned_sent)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # Filter Posative dataset and build word frequency

# In[37]:


dataset_Positive=dataset[dataset['Category'] == "positive"]


# In[38]:


dataset_Positive=dataset_Positive['Tweet']


# In[39]:


cleaned_text = ""
cleaned_sent = ""
for dataset_Tweet in dataset_Negative:
    cleaned_text=text_prepare(dataset_Tweet)
    cleaned_sent=cleaned_sent + cleaned_text


# In[40]:


word_tokens = nltk.word_tokenize(cleaned_sent)


# In[41]:


words_counts = {}
for word in word_tokens:
    if word in words_counts:
        words_counts[word] = words_counts[word] + 1
    else:
        words_counts[word] = 1 


# In[42]:


most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:20]


# In[43]:


Freq_dist_nltk=nltk.FreqDist(word_tokens)
print(Freq_dist_nltk)
sorted_d = sorted(Freq_dist_nltk.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_d[:25])
Freq_dist_nltk.plot(25, cumulative=False)


# In[44]:


wordcloud = WordCloud().generate(cleaned_sent)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # Filter Neutral dataset and build word frequency

# In[45]:


dataset_Neutral=dataset[dataset['Category'] == "neutral"]
dataset_Neutral=dataset_Neutral['Tweet']
cleaned_text = ""
cleaned_sent = ""
for dataset_Tweet in dataset_Negative:
    cleaned_text=text_prepare(dataset_Tweet)
    cleaned_sent=cleaned_sent + cleaned_text
word_tokens = nltk.word_tokenize(cleaned_sent)
words_counts = {}
for word in word_tokens:
    if word in words_counts:
        words_counts[word] = words_counts[word] + 1
    else:
        words_counts[word] = 1 
        


# In[46]:


Freq_dist_nltk=nltk.FreqDist(word_tokens)
print(Freq_dist_nltk)
sorted_d = sorted(Freq_dist_nltk.items(), key=operator.itemgetter(1),reverse=True)
print(sorted_d[:25])
Freq_dist_nltk.plot(25, cumulative=False)


# In[47]:


wordcloud = WordCloud().generate(cleaned_sent)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# # Build a model using KMeans 

# In[48]:


#Text clustering with K-means
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# In[49]:


df=df.iloc[:,0]


# In[50]:


vectorizer = TfidfVectorizer(stop_words='english')


# In[51]:


X = vectorizer.fit_transform(df)
print(X)


# **Build the clusters**

# In[52]:


true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


# **Profile the clusters: Top terms per cluster**

# In[53]:


order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),


# In[54]:


#Use model for prediction
print("\n")
print("Prediction")
Y = vectorizer.transform(["valentines"])
prediction = model.predict(Y)
print(prediction)


# # Build a model using Naiv Bayesian

# In[55]:


dataset_list_train = dataset[0:4500].values.tolist()
dataset_list_test = dataset[4500:5967].values.tolist()


# In[56]:


model=NaiveBayesClassifier(dataset_list_train)


# **Model Accuracy on trainig set**

# In[57]:


print(model.accuracy(dataset_list_train))


# **Model Accuracy on trainig set**

# In[58]:


print(model.accuracy(dataset_list_test))


# In[59]:


model.show_informative_features(5)


# In[60]:


model.classify("ios  app transport security mm need to check if my rd party network pod supports it")


# In[61]:


prob_dist = model.prob_classify("five great free apps and games for ios  august th edition  it is that time of the week again news lchbuzz")


# In[62]:


prob_dist.max()


# **Probability distribution**

# In[63]:


round(prob_dist.prob("positive"), 2)


# In[64]:


round(prob_dist.prob("neutral"), 2)


# In[65]:


round(prob_dist.prob("negative"), 2)


# # Build Lexican Based Model

# **Sentence Tokenization**

# In[66]:


tokenized_text=sent_tokenize(cleaned_tweets)


# **Word Tokenization**

# In[67]:


from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(cleaned_tweets)


# **Frequency Distribution**

# In[68]:


from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)


# In[69]:


fdist.most_common(2)


# In[70]:


fdist.plot(30,cumulative=False)
plt.show()


# **Removing Stopwords**

# In[71]:


stop_words=set(stopwords.words("english"))
print(stop_words)


# In[72]:


filtered_sent=[]
for w in tokenized_text:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_text)
print("Filterd Sentence:",filtered_sent)


# # Lexicon Normalization

# **Stemming**

# In[73]:


ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# **Lemmatization**

# **Lexicon Normalization
# performing stemming and Lemmatization**

# In[74]:


lem = WordNetLemmatizer()
stem  = PorterStemmer()
lem_words=[]
for w in tokenized_word:
    lem_words.append(lem.lemmatize(w))
print("Lemmatized Word:",lem_words)


# **POS Tagging**

# In[75]:


tokens=word_tokenize(cleaned_tweets)
print(tokens)


# In[76]:


nltk.pos_tag(tokens)


# In[77]:


dataset.Category.value_counts()


# In[78]:


Sentiment_count=dataset.groupby('Category').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Tweet'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# **Tokenizer to remove unwanted elements from out data like symbols and numbers**

# In[79]:


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cvect = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cvect.fit_transform(dataset['Tweet'])


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(text_counts, dataset['Category'], test_size=0.3, random_state=1)


# **Model Generation Using Multinomial Naive Bayes**

# In[81]:


model = MultinomialNB().fit(X_train, y_train)
predicted= model.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# **TF-IDF Vectorizer**

# In[82]:


tf=TfidfVectorizer()
text_tf= tf.fit_transform(dataset['Category'])


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(
    text_tf, dataset['Category'], test_size=0.3, random_state=123)


# **Model Generation Using Multinomial Naive Bayes**

# In[84]:


model = MultinomialNB().fit(X_train, y_train)
predicted= model.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[ ]:




