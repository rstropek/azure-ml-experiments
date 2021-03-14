import re
import unidecode
from nltk import WordNetLemmatizer

def preprocess(ingredients):
    lemmatizer = WordNetLemmatizer()
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower() #Lower - Casing
    ingredients_text = ingredients_text.replace('-', ' ') # Removing Hyphen
    words = []
    for word in ingredients_text.split():
        word = re.sub("[0-9]"," ",word) # removing numbers,punctuations and special characters
        word = re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b'), ' ', word) # Removing Units
        if len(word) <= 2: continue # Removing words with less than two characters
        word = unidecode.unidecode(word) # Removing accents
        word = lemmatizer.lemmatize(word) # Lemmatize
        if len(word) > 0: words.append(word)
    return ' '.join(words)