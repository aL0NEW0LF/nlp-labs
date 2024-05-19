import re
import string
from nltk.corpus import stopwords

def processTweet(tweet):
    '''
    Process tweet function. 
    Args:
        tweet (str): A string containing a tweet.

    Returns:
        str: The processed tweet.

    This function takes a tweet as input and performs various preprocessing steps to clean the tweet text. It removes user handles, words starting with a dollar sign, hyperlinks, hashtags, punctuations, words with 2 or fewer letters, HTML special entities, whitespace, stopwords, characters beyond the Basic Multilingual Plane (BMP) of Unicode, and converts the tweet to lowercase. The processed tweet is then returned.
    '''
    if isinstance(tweet, float):
        return str(tweet)
    # remove user handles tagged in the tweet
    tweet = re.sub('@[^\s]+','',tweet)
    # remove words that start with th dollar sign    
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    tweet = re.sub(r'(?:^|[\s,])([\w-]+\.[a-z]{2,}\S*)\b','',tweet)
    # remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # remove all kinds of punctuations and special characters
    punkt = string.punctuation + r'''`‘’)(+÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”،.”…“–ـ”.°ा'''
    tweet = tweet.translate(str.maketrans('', '', punkt))
    # remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # remove stopwords
    tweet = re.sub(r'\b('+ '|'.join(stopword for stopword in stopwords.words('english'))+ r')\b', '', tweet)
    # remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ')
    tweet = tweet.rstrip(' ')
    # remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uffff')
    tweet = re.sub(r'([^\u1F600-\u1F6FF\s])','', tweet)
    # lowercase
    tweet = tweet.lower()
    # remove extra spaces
    tweet = re.sub(r'[\s]{2, }', ' ', tweet)
    
    return tweet
