# Libraries used
import time
import glob
import pandas as pd 
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords # Import the stop word list
import re
import requests
from bs4 import BeautifulSoup # Use bs4 to remove html tags or markup
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle

print('All libraries are imported successfully!')

# Import data
print('\nData Importing.............')

os.chdir("D:\\TopicData\\Adidas")
path =r'D:\\TopicData\\Adidas'
filenames = glob.glob(path + "/*.csv")
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, header=0, delimiter=",", encoding = "ISO-8859-1"))

Train = pd.concat(dfs, ignore_index=True)

print('\nShuffle dataset to make robust...')
Train.iloc[np.random.permutation(len(Train))]
Train.reset_index(drop=True)

print('\nTraining Data imported with dimensions:', Train.shape)

#Data Preprocessing

print("\nRemoving NA values...")
Train = Train.replace(np.nan,' ', regex=True)

print("\nDividing data into train and test set...")
msk = np.random.rand(len(Train)) < 0.95

train = Train[msk]
test = Train[~msk]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Data Cleaning

def tweet_to_words( raw_tweet ):
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_tweet).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. Make list of stopwords
    stops = set(stopwords.words("english"))      
    stops.update(['rt', 'youtube', 'https', 'co', 'via', 'always', 'another', 'coming', 'could', 'days', 'everyone', 'get', 'give', 'going', 'liked', 'pm', 'say', 'th', 'still', 'vs', 'would', 'years'])            
    # 
    # 5. Remove stop words
    meaningful_tweet = [w for w in words if not w in stops]  
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_tweet ))

# Get the number of tweets based on the dataframe column size
num_tweets = train["Tweet"].size

# Initialize an empty list to hold the clean tweets
clean_train_tweets = []

print("\nCleaning and parsing the training set tweets...")
startTime = time.time()

for i in range( 0, num_tweets ):
    # If the index is evenly divisible by 100000, print a message
    if( (i+1)%100000 == 0 ):
        print("Tweet %d of %d\n" % ( i+1, num_tweets ))                                                                   
    clean_train_tweets.append( tweet_to_words( train["Tweet"][i] ))
    
print('\nCleaning of all tweets have been done in ', (time.time()-startTime)/60, 'Minutes.')
# Object Serialization
##pickle_in = open('clean_train_tweets.pickle','wb')
##pickle.dump(clean_train_tweets, pickle_in)
##pickle_in.close()
##pickle_in = open('clean_train_tweets.pickle','rb')
##clean_train_tweets = pickle.load(pickle_in)
print('\nWe can use pickling to get cleaned trained tweets (clean_train_tweets.pickle, SGDClassifier.pickle).')

# Generate a word cloud image
text = ' '.join(clean_train_tweets)
wordcloud = WordCloud().generate(text)

# Number of words in train dataset
num1 = 0
for item in clean_train_tweets:
    num = len(item.split())
    num1 = num1 + num
    
print("\n",num1, "number of words have used to build the word cloud.")

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')

# Initialize the "HashingVectorizer" object, which is scikit-learn's
# bag of words tool with max_features  
vectorizer = HashingVectorizer(analyzer = "word", \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             ngram_range = (1, 2),\
                             n_features = 2**20)

train_data_features = vectorizer.fit_transform(clean_train_tweets)

topwords_vector = CountVectorizer(analyzer = "word",   \
                                  tokenizer = None,    \
                                  preprocessor = None, \
                                  stop_words = None,   \
                                  max_features = 150)

print("Below are the top 50 words commonly occured: \n")
train_data_features1 = topwords_vector.fit_transform(clean_train_tweets)
vocab = topwords_vector.get_feature_names()
print(vocab)

print('\nWe have created  document term matrix (DTM) with dimensions ', train_data_features.shape, ' and we have target variable with dimensions ', train['Tweet'].shape)

#Batch Processing
print("\nDoing batch processing...")

numtrainingpoints = len(train)
def iter_minibatches(chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < numtrainingpoints:
        chunkrows = chunkstartmarker+chunksize
        try:
            X_chunk, Y_chunk = train_data_features[chunkstartmarker:chunkrows],train["Subtopic"][chunkstartmarker:chunkrows]
            yield X_chunk, Y_chunk
            chunkstartmarker += chunksize
        except IndexError:
            print("\nAll batches are processed!!!")
            break

print('\n==========================================================================================')

# Using generator to create small batches from sparse matrix

print("\nDividing data into small chunks with size 100K...")
batcherator = iter_minibatches(chunksize=100000)

clf = SGDClassifier(loss='modified_huber', random_state=1, n_iter=1, n_jobs = -1, penalty='l1')

#clf = MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive', hidden_layer_sizes=(50,), random_state=1, max_iter=1)

classes = np.unique(train['Subtopic'])
BatchNumber = 1
for X_chunk, Y_chunk in batcherator:
    Y_chunk = Y_chunk.reset_index()
    clf.partial_fit(X_chunk, Y_chunk['Subtopic'], classes = classes)
    print('\n', BatchNumber ,'Batch Processed')
    BatchNumber += 1

# Processing Test data
# Create an empty list and append the clean reviews one by one
num_tweets_test = len(test["Tweet"])
clean_test_tweets = [] 

print("\nCleaning and parsing the test set tweets...")
for i in range(0,num_tweets_test):
    if( (i+1) % 100000 == 0 ):
        print("Tweet %d of %d\n" % (i+1, num_tweets_test))
    clean_tweet = tweet_to_words( test["Tweet"][i] )
    clean_test_tweets.append( clean_tweet )
print("\nTest data has been cleaned successfully! Hurray!")

test_data_features = vectorizer.transform(clean_test_tweets)

print("\nPredicting on test data and storing in file output...")
result = clf.predict(test_data_features)

# Copy the results to a pandas dataframe with an "Tweet" column and
# a "Subtopic" column
output = pd.DataFrame( data={"Tweet":test["Tweet"], "Subtopic":result} )

print('\nWe have received accuracy of', accuracy_score(test['Subtopic'], output['Subtopic'])*100) 

print('\nBelow is the confusion matrix for classifier:\n')
confusion_matrix(test['Subtopic'], output['Subtopic'])

# Use pandas to write the comma-separated output file
# output.to_csv( "Predicted_topics.csv",sep=',', encoding='utf-8', index=False)

##print('\n Predicted output as follows:')
##output.head()

##a = (output.groupby('Subtopic').size() / output.groupby('Subtopic').size().sum())*100
##a = pd.DataFrame({'Subtopic':a.index, 'Percentage':a.values})
##a = a.sort(['Percentage'], ascending=False)
##a = a[["Subtopic", "Percentage"]]
##a = a.reset_index(drop=True)

print('\nPrediction of maximum likelihood topics is completed now!')

print('\nChecking sentiments of tweets related to each stopic...')
print('\nFor 6000 thousand tweets it will approximately takes 1 hour to find sentiment......it is slow but soon building fast sentiment analysis model...;)')

#####################################################################

output = output[0:133929]
s = []
c = []
exceptMe = 0

tweet = output['Tweet']

for i in tweet:
    time.sleep(0.300)
    response = requests.post("https://text-sentiment.p.mashape.com/analyze",
                             headers={
                                 "X-Mashape-Key": "W0cPU4Age0mshP3aCRzwx3ORgPh4p1t4jmdjsnEtljSMYauL6n",
                                 "Content-Type": "application/x-www-form-urlencoded",
                                 "Accept": "application/json"
                                 },
                             params={
                                 "text": i
                                 }
                             )
    result = response.text
    result = result.replace('"', '')
    result = result.replace(',', ' ')

    try:
        posSent = re.search('pos:(.+?)neg', result)
        posSent = posSent.group(1)

        negSent = re.search('neg:(.+?)mid', result)
        negSent = negSent.group(1)

        neutralSent = re.search('mid:(.+?)pos_percent', result)
        neutralSent = neutralSent.group(1)

        posConf = re.search('pos_percent:(.+?)%', result)
        posConf = posConf.group(1)

        negConf = re.search('neg_percent:(.+?)%', result)
        negConf = negConf.group(1)

        neutralConf = re.search('mid_percent:(.+?)%', result)
        neutralConf = neutralConf.group(1)

    except AttributeError:
        posSent = 1
        posConf = 50
        negSent = 0
        negConf = 0
        neutralSent = 0
        neutralConf = 0
        exceptMe += 1

    sentiment = [posSent, negSent, neutralSent]
    confidence = [posConf, negConf, neutralConf]

    if(int(sentiment[0])!= 0):
        s.append('Positive')
        c.append(confidence[0])
    elif(int(sentiment[1]) != 0):
        s.append('Negative')
        c.append(confidence[1])
    elif(int(sentiment[2])!= 0):
        s.append('Neutral')
        c.append(confidence[2])
    
print("\nWarning: I have added ",exceptMe," exceptions.")
df = pd.DataFrame( data={"Subtopic": output["Subtopic"], "Tweet": output["Tweet"], "Sentiment": s, "Confidence": c})

seriesA = (df.groupby('Subtopic').size() / df.groupby('Subtopic').size().sum())*100
seriesA = pd.DataFrame({'Subtopic':seriesA.index, 'Percentage':seriesA.values})

##print('\nTopic distribution in percentages as follows:\n')
##print(seriesA.head(10))

topics = np.unique(df[['Subtopic']])

abc = []
posList = []
negList = []
neutral = []

for j in topics:
    abc = df.loc[df['Subtopic'] == j]
    totalC = len(abc)
    posC = (len(abc.loc[abc['Sentiment'] == 'Positive'])/len(abc))*100
    negC = (len(abc.loc[abc['Sentiment'] == 'Negative'])/len(abc))*100
    neutralC = (len(abc.loc[abc['Sentiment'] == 'Neutral'])/len(abc))*100

    posList.append(posC)
    negList.append(negC)
    neutral.append(neutralC)
        
df_final = pd.DataFrame( data={"Subtopic": seriesA["Subtopic"], "Percentage": seriesA["Percentage"], "Positive_Sent": posList, "Negative_Sent": negList,"Neutral_Sent": neutral})
df_final = df_final[['Subtopic', 'Percentage', 'Positive_Sent', 'Negative_Sent', 'Neutral_Sent']] 
df_final = df_final.sort(['Percentage'], ascending=False)
df_final = df_final.reset_index(drop=True)

print(df_final.head(11))

df_final.to_csv("final_result.csv", sep=',', encoding='utf-8', index=False)

print("\nPlease check CSV file final_result.csv in set directory for results. Enjoy Machine Learning. Thanks! Mayur Mane.")
