# **NATURAL LANGUAGE PROCESSING - LAB 13**

**Done by:** MOHAMED AMINE FAKHRE-EDDINE

**Supervised by:** Pr. Lotfi EL AACHAK

**Date:** 19/05/2024

## **1. Lab report**
### **1.1. Language Modeling - Regression**
In this part of the lab, we learned about language modeling using regression, to predict the score of an answer for a specific question.

#### **1.1.1. Data Preprocessing**
We started by preprocessing the data, which consisted of tokenizing the text, removing stopwords, and converting the text to lowercase.

#### **1.1.2. Tokenization**
Tokenization is the process of splitting text into individual words or tokens. We used the `nltk` library to tokenize the text.

#### **1.1.3. Word Embeddings**
We trained word embeddings using the `Word2Vec` model from the `gensim` library. Word embeddings are dense vector representations of words that capture semantic relationships between words.

So we each question with multiple answers, we trained a cbow model to have a vector representation of each word, then we aggregated the vectors of the words in the question to have a vector representation of the answers.

We used a custom function to fine-tune the model since regression models are pretty sensitive to the vector size of the word2vec model we would dbe using.

#### **1.1.4. Regression**
We used regression to predict the score of an answer given a question. 

After finding the best vector size for each regression model notably `LinearRegression`, `SVR`, `DecisionTreeRegressor`, we evaluated the models using the mean squared error and the R2 score.

### **1.2. Language Modeling - Classification**
In this part of the lab, we learned about language modeling using classification, to prediction the sentiment of a tweet.

#### **1.2.1. Data Preprocessing**
Same as the regression part, we started by extensively preprocessing the data, which consisted of removing user handles, words starting with a dollar sign, hyperlinks, hashtags, punctuations, words with 2 or fewer letters, HTML special entities, whitespace, stopwords, characters beyond the Basic Multilingual Plane (BMP) of Unicode, and converting the tweet to lowercase.

#### **1.2.2. Tokenization**
Same as the regression part, we used the `nltk` library to tokenize the text.

#### **1.2.3. Word Embeddings**
Same as the regression part, we trained word embeddings using the `Word2Vec` model from the `gensim` library. 

#### **1.2.4. Classification**
We used classification to predict the sentiment of a tweet.

After finding the best vector size for each classification model notably `LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `AdaBoostClassifier`, `LogisticRegressionCV` we evaluated the models using the accuracy score, and F1 score.

## **2. What have we learned?**
Regression and classification are two common techniques used in language modeling to predict continuous and discrete values, respectively. In this lab, we applied regression to predict the score of an answer given a question and classification to predict the sentiment of a tweet. We used word embeddings to represent words as dense vectors and trained regression and classification models on these vectors. We evaluated the models using metrics such as mean squared error, R2 score, accuracy score, and F1 score.

Additionally, we learned that regression models are pretty sensitive to the vector size of the word2vec model we would be using, pretty sensitive to the smallest change in the vector size, in contrast to classification models, which are more robust to changes in the vector size, so a small change in the vector size would not affect the performance of the model that much, that's why fine-tuning word2vec model for classification would take so much more time.