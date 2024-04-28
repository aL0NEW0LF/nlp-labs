import argparse

from nltk.stem.isri import ISRIStemmer
import sys


# Arabic light stemming for Arabic text
# takes a word list and perform light stemming for each Arabic words
def light_stem(text: str | list):
    if isinstance(text, str):
        words = text.split()
    elif isinstance(text, list):
        words = text
    else:
        sys.exit("Invalid input type, must be either a string or a list of words")
    
    result = list()
    stemmer = ISRIStemmer()
    for word in words:
        word = stemmer.norm(word, num=1)      # remove diacritics which representing Arabic short vowels
        if not word in stemmer.stop_words:    # exclude stop words from being processed
            word = stemmer.pre32(word)        # remove length three and length two prefixes in this order
            word = stemmer.suf32(word)        # remove length three and length two suffixes in this order
            word = stemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
            word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
        result.append(word)
    return result