import re

from matplotlib import units

def hundreds_sum(hundreds: list):
    hundreds_index = hundreds.index(100) if 100 in hundreds else -1
    total = 0

    if hundreds_index > -1:
        if hundreds_index == 0:
            total += 100
        else:
            total += hundreds[hundreds_index - 1] * 100
        hundreds = hundreds[hundreds_index + 1:]

    if len(hundreds) > 0:
        total += sum(hundreds)

    return total
            
def parse_number(number_sentence):
    number_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1000000
    }
    
    number_sentence = number_sentence.replace('-', ' ')
    number_sentence = number_sentence.lower()

    if(number_sentence.isdigit()):
        return int(number_sentence)

    split_words = number_sentence.strip().split()

    clean_numbers = []

    for word in split_words:
        if word in number_map:
            clean_numbers.append(number_map[word])

    if len(clean_numbers) == 0:
        raise ValueError("Not valid number")

    if clean_numbers.count(1000) > 1 or clean_numbers.count(1000000) > 1:
        raise ValueError("1000 or 1000000 can only appear once in a number") 
    
    million_index = clean_numbers.index(1000000) if 1000000 in clean_numbers else -1
    thousand_index = clean_numbers.index(1000) if 1000 in clean_numbers else -1

    if (thousand_index > -1 and (thousand_index < million_index)):
        raise ValueError("Thousand can't appear before million in a number")
    
    millions = []
    thousands = []
    
    if million_index > -1:
        if million_index == 0:
            millions.append(1000000)
        else:
            millions = clean_numbers[:million_index]
            clean_numbers = clean_numbers[million_index + 1:]
    
    if thousand_index > -1:
        if thousand_index == 0:
            thousands.append(1000)
        else:
            thousands = clean_numbers[:thousand_index - million_index - 1]
            clean_numbers = clean_numbers[thousand_index - million_index:]

    return hundreds_sum(millions) * 1000000 + hundreds_sum(thousands) * 1000 + hundreds_sum(clean_numbers)

def print_bill(bill: list):
    bill = [['Product', 'Quantity', 'Unit Price']] + bill

    quantity_column = max([len(str(product[0])) for product in bill])
    product_column = max([len(product[1]) for product in bill])
    unit_price_column = max([len(str(product[2])) for product in bill])

    bill.pop(0)

    print(f"{'Product':<{product_column}} | {'Quantity':<{quantity_column}} | {'Unit Price':<{unit_price_column}} | Total Price")
    print('-' * (product_column + quantity_column + unit_price_column + 21))

    for product in bill:
        print(f"{product[1]:<{product_column}} | {product[0]:<{quantity_column}}  | {product[2]:<{unit_price_column}} | {product[0] * product[2]}")

def generate_bill(text: str):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
                 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
                 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                 "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]    
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million']
    pattern = r"((?:" + '|'.join(numbers) + r"|\d)(?:\s(?:" + '|'.join(numbers) + r"|\d|and))*)(.*?)(\d+[\.|\,]?\d*)\b\s*(\$|dollar)"
    units = ['kg', 'kilogram', 'g', 'gram', 'l', 'liter', 'ml', 'milliliter', 'mg', 'milligram', 'lb', 'pound', 'oz', 'ounce', 'gallon', 'quart', 'pint', 'cup', 'tablespoon', 'teaspoon', 'ton', 't', 'kilos', 'kilograms', 'grams', 'liters', 'milliliters', 'milligrams', 'pounds', 'ounces', 'gallons', 'quarts', 'pints', 'cups', 'tablespoons', 'teaspoons', 'tons', 'ts', 'tbs', 'tsp', 'kg.', 'g.', 'l.', 'ml.', 'mg.', 'lb.', 'oz.', 'gal.', 'qt.', 'pt.', 'cup.', 'tbsp.', 'tsp.', 't.', 'kgs', 'kgs.', 'kgs.', 'gms', 'gms.', 'gms.', 'lts', 'lts.', 'lts.', 'mls', 'mls.', 'mls.', 'mgs', 'mgs.', 'mgs.', 'lbs', 'lbs.', 'lbs.', 'ozs', 'ozs.', 'ozs.', 'gals', 'gals.', 'gals.', 'qts', 'qts.', 'qts.', 'pts', 'pts.', 'pts.', 'cups', 'cups.', 'tbsps', 'tbsps.', 'tbsps.', 'tsps', 'tsps.', 'tsps.', 'tons.', 'ts.', 'tbs.', 'tsp.', 'kgs.', 'gms.', 'lts.', 'mls.', 'mgs.', 'lbs.', 'ozs.', 'gals.', 'qts.', 'pts.', 'cups.', 'tbsps.', 'tsps.', 'tons.', 'ts.', 'tbs.', 'tsp.', 'kgs.', 'gms.', 'lts.', 'mls.', 'mgs.', 'lbs.', 'ozs.', 'gals.', 'qts.', 'pts.', 'cups.', 'tbsps.', 'tsps.', 'tons.', 'ts.', 'tbs.', 'tsp.', 'kgs.', 'gms.', 'lts.', 'mls.', 'mgs.', 'lbs.', 'ozs.', 'gals.', 'qts.', 'pts.', 'cups.', 'tbsps.', 'tsps.', 'tons.', 'ts.', 'tbs.', 'tsp.', 'kgs.', 'gms.', 'lts.', 'mls.', 'mgs.', 'lbs.', 'ozs.', 'gals']

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    for stop_word in stop_words:
        insensitive_stop_word = re.compile(re.escape(' ' + stop_word + ' '), re.IGNORECASE)
        text = insensitive_stop_word.sub(' ', text)
    
    matches = re.findall(pattern, text)
    
    sc = set(units)

    matches = [[parse_number(match[0].strip()), ' '.join([word for word in match[1].split() if word not in sc]), float(match[2].replace(',', '.'))] for match in matches]

    return matches