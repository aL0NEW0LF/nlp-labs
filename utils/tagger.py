import re

def tag(text: str):
    words = text.split()
    result = []
    conjunctions_noun = ["ليت", "و", "لعل", "كأن", "إِنَّ", "تحت", "وراء", "حيث", "دون", "حين", "صباح", "ظهر",
                            "أمام", "إما", "إلا", "أو", "بعد", "ب", "فوق", "كل", "لي", "لها", "لنا", "لهم", "لك", "له",
                            "هن", "هما", "أي", "يا", "أيتها", "أيها", "هيا", "أيا", "على", "إلى", "عن", "من", "في"]
    conjunctions_verb = ["أن", "لن", "كي", "حتى", "لم", "لما", "لا", "إن", "ما", "مهما", "متى", "أينما", "حيثما",
                            "قد"]
    conjunctions_particle = ["أياي", "ايانا", "اياك", "اياكما", "اياكم", "اياكن", "اياه", "اياها", "اياهم",
                                "اياهن", "اياهما"]
    kan = ["ليس", "بات", "أمسى", "ظل", "أضحى", "أصبح", "صار", "كان"]
    nounprefix = ["ال", "لل", "فال", "كال", "بال"]  # check first
    verbprefix = ["سي", "سن", "ست", "سأ"]

    # to do: add suffixes

    verbpatterns = ["ا..و..", "ا.ت..", "يت...", "يت.ا..", "است...", "ا..ا.", "يست...", "تم...", "ت.و..", "ي...ون",
                    "ت...ان", "ي...ان", "ت...ون", "يت...ون", "يت.ا..ون", "يست...ون", "يت...ان", "يت.ا..ان", "أ...ت", "يست...ان", "تت...ون", "تت.ا..ون", "تست...ون"]
    nounpatterns = ["م...", "م..ا.", "..ا.ة", "م...ة", "...ى", "..ي.", "...ة$"]

    for word in words:
        if word in conjunctions_noun:
            result.append("P")
            result.append("N")
            continue
        elif word in conjunctions_verb:
            result.append("P")
            result.append("V")
            result.append("N")
            continue
        elif word in conjunctions_particle:
            result.append("P")
            result.append("P")
            continue
        elif word in kan:
            result.append("V")
            result.append("N")
            continue
        elif word == "و":
            result.append("P")
            result.append(result[i - 1])
            continue
        else:
            flag = True
            if flag:
                for prefix in nounprefix:
                    if word.startswith(prefix):
                        result.append("N")
                        flag = False
                        break

            if flag:
                for prefix in verbprefix:
                    if word.startswith(prefix):
                        result.append("V")
                        result.append("N")
                        flag = False
                        break

            if flag:
                for pattern in verbpatterns:
                    if len(pattern) == len(word):
                        if re.match(pattern, word) and len(pattern) == len(word):
                            result.append("V")
                            result.append("N")
                            flag = False
                            break

            if flag:
                for pattern in nounpatterns:
                    if len(pattern) == len(word):
                        if re.match(pattern, word) and len(pattern) == len(word):
                            result.append("N")
                            flag = False
                            break
            if flag:
                result.append("N")

    return result

def dumbPOSTagger(text: str):
    # Rule Based Approach for Arabic Part of Speech Tagging
    """ 
    POS Tagger for Arabic Text.

    Args:
        text: str -> Arabic text to be tagged.
    Rules:
        Rule 1: the following prefixes "كال"," بال"," فال"," وال "if it comes in the beginning of a word map it refers to Noun class.
        Rule 2: the following suffixes in comes it if" ائي","ائك","ائه","اؤك","اؤه","اءك","اءه","هما","كما" the end of a word map itrefers to Noun Class.
        Rule 3: the following prefixes in comes it if" سي","ست","سن","سأ","سا","ال","أل","لن","لت","لي" the end of a word map itrefers toVerb class.
        Rule 4: if the following suffixes " و" ,"ن" ,"ا" ,"ك" ,"ه" ,"ي"  comes in the end of a word map it refers to Verb class.
        Rule 5: if the word has the pattern (فعاء ,فعول ,فعلى)map it to Noun class.
        Rule 6: if the word ends with "ات "map it to Noun class.
        Rule 7: if the word end with "ين ,ون"but starts with "ي "or "ن "map the word to Verb class.
        Rule 8: if the word ends with "ين ,ون "and doesn’t start with "ي "or "ن "map the word to Noun class.
        Rule 9: if the word has the pattern (مفاعل,مفعيل,مفعال,مفتعل,منفعل,مفعول,متفعل,مفعلل) map it to Noun class.
        Rule 10: if the word has the pattern (مفاعيل ,أفاعيل ,فعاليل فواعيل)map it to Noun class.
        Rule 11: if the word has the pattern (افعوعل ,استفعل)map it to Verb class.
    """
    words = text.split()
    tags = {}

    conjunctions_noun = ["ليت", "و", "لعل", "كأن", "إِنَّ", "تحت", "وراء", "حيث", "دون", "حين", "صباح", "ظهر", "أمام", "إما", "إلا", "أو", "بعد", "ب", "فوق", "كل", "لي", "لها", "لنا", "لهم", "لك", "له", "هن", "هما", "أي", "يا", "أيتها", "أيها", "هيا", "أيا", "على", "إلى", "عن", "من", "في"]
    conjunctions_verb = ["أن", "لن", "كي", "حتى", "لم", "لما", "لا", "إن", "ما", "مهما", "متى", "أينما", "حيثما", "قد"]
    conjunctions_particle = ["أياي", "ايانا", "اياك", "اياكما", "اياكم", "اياكن", "اياه", "اياها", "اياهم", "اياهن", "اياهما"]
    kan = ["ليس", "بات", "أمسى", "ظل", "أضحى", "أصبح", "صار", "كان"]

    # TODO: Add Suffixes

    for word in words:
        if word in conjunctions_noun:
            tags[word] = 'NOUN'
        elif word in conjunctions_verb:
            tags[word] = 'VERB'
        elif word in conjunctions_particle:
            tags[word] = 'PART'
        elif word in kan:
            tags[word] = 'VERB'
        elif word == "و":
            tags[word] = 'CONJ'
        elif word.startswith('ال') or word.startswith('كال') or word.startswith('بال') or word.startswith('فال') or word.startswith('وال'):
            tags[word] = 'NOUN'
        elif word.endswith('ائي') or word.endswith('ائك') or word.endswith('ائه') or word.endswith('اؤك') or word.endswith('اؤه') or word.endswith('اءك') or word.endswith('اءه') or word.endswith('هما') or word.endswith('كما'):
            tags[word] = 'NOUN'
        elif word.startswith('سي') or word.startswith('ست') or word.startswith('سن') or word.startswith('سأ') or word.startswith('سا') or word.startswith('ال') or word.startswith('أل') or word.startswith('لن') or word.startswith('لت') or word.startswith('لي'):
            tags[word] = 'VERB'
        elif word.endswith('و') or word.endswith('ن') or word.endswith('ا') or word.endswith('ك') or word.endswith('ه') or word.endswith('ي'):
            tags[word] = 'NOUN'
        elif word.endswith('ات'):
            tags[word] = 'NOUN'
        elif word.endswith('ين') or word.endswith('ون'):
            if word.startswith('ي') or word.startswith('ن'):
                tags[word] = 'VERB'
            else:
                tags[word] = 'NOUN'
        elif re.match(r'\b([\u0621-\u064A\u06A9-\u06C1]{2}\u0627{1}\u0621{1}|[\u0621-\u064A\u06A9-\u06C1]{3}\u0649{1})\b', word):
            tags[word] = 'NOUN'
        elif re.match(r'\b([\u0621-\u064A\u06A9-\u06C1]{2}\u0648{1}[\u0621-\u064A\u06A9-\u06C1]{1})\b', word):
            tags[word] = 'ADJ'
        elif re.match(r'\b(\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}|\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u0645{1}\u0646{1}[\u0621-\u064A\u06A9-\u06C1]{3}|\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0648{1}[\u0621-\u064A\u06A9-\u06C1]{1}|\u0645{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}|\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{2}[\u0621-\u064A\u06A9-\u06C1]{1}\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{1}|\u0623{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{1}|[\u0621-\u064A\u06A9-\u06C1]{2}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{1}|[\u0621-\u064A\u06A9-\u06C1]{1}\u0648{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{1}|[\u0621-\u064A\u06A9-\u06C1]{2}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0629{1}|\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0629{1}|[\u0621-\u064A\u06A9-\u06C1]{3}\u0629{1}|[\u0621-\u064A\u06A9-\u06C1]{3}\u0649{1}|[\u0621-\u064A\u06A9-\u06C1]{2}\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{1})\b', word):
            # nounpatterns = ["م...", "م..ا.", "..ا.ة", "م...ة", "...ى", "..ي.", "...ة$"]
            tags[word] = 'NOUN'
        elif re.match(r'\b(\u0627{1}\u0633{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}|\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0648{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u062A{1}\u0633{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0648{1}\u0646{1}|\u064A{1}\u0633{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0648{1}\u0646{1}|\u064A{1}\u0633{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0627{1}\u0646{1}|\u062A{2}[\u0621-\u064A\u06A9-\u06C1]{3}\u0648{1}\u0646{1}|\u062A{2}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0648{1}\u0646{1}|\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0627{1}\u0646{1}|\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0627{1}\u0646{1}|\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0648{1}\u0646{1}|\u064A{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0648{1}\u0646{1}|\u064A{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0648{1}\u0646{1}|\u064A{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0627{1}\u0646{1}|\u064A{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0627{1}\u0646{1}|\u0623{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u062A{1}|\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0648{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u064A{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}|\u064A{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{2}\u0627{1}[\u0621-\u064A\u06A9-\u06C1]{1}|\u064A{1}\u0633{1}\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{3}|\u062A{1}\u0645{1}[\u0621-\u064A\u06A9-\u06C1]{3}|\u062A{1}[\u0621-\u064A\u06A9-\u06C1]{1}\u0648{1}[\u0621-\u064A\u06A9-\u06C1]{2}|\u064A{1}[\u0621-\u064A\u06A9-\u06C1]{3}\u0648{1}\u0646{1})\b', word):
            # verbpatterns = ["ا..و..", "ا.ت..", "يت...", "يت.ا..", "است...", "ا..ا.", "يست...", "تم...", "ت.و..", "ي...ون", "ت...ان", "ي...ان", "ت...ون", "يت...ون", "يت.ا..ون", "يست...ون", "يت...ان", "يت.ا..ان", "أ...ت", "يست...ان", "تت...ون", "تت.ا..ون", "تست...ون"]
            tags[word] = 'VERB'
        else:
            tags[word] = 'NOUN'

    return tags