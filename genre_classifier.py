# Classifier Notes

# Using Brown Corpus to classify the genre of the entered text

# Will be testing machine with 10 documents,
#  consisting of several news and research
#   articles, novels, and religious texts
#    (initial test will be to classify adventure, lore, and romance)

# genres: adventure, belles_lettres, editorial, fiction,
#  government, hobbies, humor,
#   learned, lore, mystery, news, religion, reviews, romance, science_fiction

# for all stored documents of a given genre
#  remove noise terms
#   obtain the percentages of the words used and store them
#    (naive bayes)

# for all inputted documents
#  tokenize and create a bag of word to obtain the percentages of them

# cases will be assigned to what data is being 
# 	worked on in noted section

from __future__ import division
import nltk
from nltk.corpus import brown

# all words per genre (def: case 1)
romance_words = brown.words(categories='romance')
adventure_words = brown.words(categories='adventure')

# all words per book per genre (def: case 2)
romance_books = [brown.words(fileid) for fileid in brown.fileids('romance')]
adventure_books = [brown.words(fileid) for fileid in brown.fileids('adventure')]

# combo of words per book per genre
all_books = romance_books + adventure_books

# frequent words (case 2)
romance_freq = nltk.FreqDist(w.lower() for w in romance_books)
adventure_freq = nltk.FreqDist(w.lower() for w in adventure_books)
# shorten frequent word list
rshort_freq = list(romance_freq)[:2000]
ashort_freq = list(adventure_freq)[:2000]

# extract stopwords from lists (case 2)
rshort_freq.extend([words.lower() for words in romance_words 
                 if not words.isnumeric() and  # Remove numbers
                  words.isalnum() and  # Remove punctuation
                  len(words) > 1 and  # Remove single characters
                  words not in all_stopwords])  # Remove stopwords

ashort_freq.extend([words.lower() for words in romance_words
                 if not words.isnumeric() and  # Remove numbers
                  words.isalnum() and  # Remove punctuation
                  len(words) > 1 and  # Remove single characters
                  words not in all_stopwords])  # Remove stopwords

# frequency of words in a specific list (useless function)
# def word_frequency(list):
#     return nltk.FreqDist(brown.words(list))

# stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('adventure', 'romance', 'page', 'chapter', 'said'))
all_stopwords = default_stopwords | custom_stopwords

# all words for all books (for cleaning)
all_words = list()
train_data = list()

all_words.extend([words.lower() for words in romance_words
                 if not words.isnumeric() and  # Remove numbers
                  words.isalnum() and  # Remove punctuation
                  len(words) > 1 and  # Remove single characters
                  words not in all_stopwords])  # Remove stopwords
all_words.extend([words.lower() for words in adventure_words
                 if not words.isnumeric() and  # Remove numbers
                  words.isalnum() and  # Remove punctuation
                  len(words) > 1 and  # Remove single characters
                  words not in all_stopwords])  # Remove stopwords

train_data = list()

# creating test and training set

# test print
# print(all_words)
