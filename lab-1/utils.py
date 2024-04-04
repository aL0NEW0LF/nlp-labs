from typing import List
from nltk.tokenize import word_tokenize

def tagsDict(taggedSentences: List[str], method: str, removeNonEntities: bool = False):
    """ 
    Extracts tags from tagged sentences.
    Args:
        - taggedSentences (List[str]): List of tagged sentences.
        - method (str): Method used for tagging. can be 'pos' or 'ner'.
    """
    validMethods = ['pos', 'ner']
    if method not in validMethods:
        raise ValueError(f"Invalid method. method must be one of {validMethods}")
    if not isinstance(taggedSentences, list) or not all(isinstance(s, str) for s in taggedSentences):
        raise ValueError("taggedSentences must be a list of strings.")
    if not isinstance(removeNonEntities, bool):
        raise ValueError("removeNonEntities must be a boolean.")
    if method == 'pos' and removeNonEntities:
        raise ValueError("removeNonEntities can only be used with method='ner'.")

    taggedTokens = []

    for p in taggedSentences:
        taggedTokens.append(word_tokenize(p))

    if method == 'pos':
        for p in taggedTokens:
            IndexToRemove = []
            for i in range(len(p)):
                if "/" not in p[i]:
                    p[i+1] = p[i] + p[i+1]
                    IndexToRemove.append(i)
            for i in sorted(IndexToRemove, reverse=True):
                del p[i]

    TagsDictsList = []

    for p in taggedTokens:
        posDict = {}
        for w in p:
            w = w.split('/')
            posDict[w[0]] = w[1]
        TagsDictsList.append(posDict)

    if removeNonEntities:
        for p in TagsDictsList:
            for w in list(p.keys()):
                if p[w] == 'O':
                    p.pop(w)

    return TagsDictsList