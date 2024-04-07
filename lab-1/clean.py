import re
import string
import argparse
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida

                         """, re.VERBOSE)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


parser = argparse.ArgumentParser(description='Pre-process arabic text (remove '
                                             'diacritics, punctuations, and repeating '
                                             'characters).')

parser.add_argument('-i', '--infile', type=argparse.FileType(mode='r', encoding='utf-8'),
                    help='input file.', required=True)
parser.add_argument('-o', '--outfile', type=argparse.FileType(mode='w', encoding='utf-8'),
                    help='out file.', required=True)

def cleaningPipeline(text: str):
    content = text

    content = remove_punctuations(content)
    print("Puctuations removed!")
    content = remove_diacritics(content)
    print("Text discretized!")
    content = normalize_arabic(content)
    print("Text normalized!")

    contentParagraphs = content.split('\n')
    print("Text tokenized!")
    contentParagraphs = list(filter(None, contentParagraphs))

    res = []
    for ele in contentParagraphs:
        if ele.strip():
            res.append(ele)

    contentParagraphs = res

    del res

    for p in contentParagraphs:
        p = sent_tokenize(p)
    print("Paragraphs tokenized!")

    stop_words = set(stopwords.words('arabic'))
    contentWords = []

    for p in contentParagraphs:
        words = word_tokenize(p)
        words = [w for w in words if not w in stop_words]
        contentWords.append(words)
    print("Words tokenized without stop words!")
    print("Cleaning completed!")
    
    return contentWords