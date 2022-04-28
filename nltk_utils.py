import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer() # create stemmer

def tokenize(sentence):
    """Splits the sentence into an array of words,
    a token can consist of a word or a punctuation mark, or number"""
    return nltk.word_tokenize(sentence)

def stem(word):
    """
        stemming = this finds the root form of the word
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    returns bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initailize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag


"""Trying out tokenization of sentences"""
a = "Hi there, what can I do for you?"
# print(a)
# a = tokenize(a)
# print(a)

"""Trying out stemming of words"""
# words = ["organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)