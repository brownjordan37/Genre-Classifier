# Classifier Notes

# Using Brown Corpus to classify the genre of the entered text

# Will be testing machine with 10 documents, consisting of several news and research
# articles, novels, and religious texts
# (initial test will be to classify adventure, lore, and romance)

# genres: adventure, belles_lettres, editorial, fiction, government, hobbies, humor,
# learned, lore, mystery, news, religion, reviews, romance, science_fiction

# for all stored documents of a given genre
# remove noise terms
# obtain the percentages of the words used and store them

# for all inputted documents 
# tokenize and create a bag of word to obtain the percentages of them

from __future__ import division
import nltk
from nltk.corpus import brown

# data
romance_words = brown.words(categories='romance')
adventure_words = brown.words(categories='adventure')

# all words per book for genres
romance_books = [brown.words(fileid) for fileid in brown.fileids('romance')]
adventure_books = [brown.words(fileid) for fileid in brown.fileids('adventure')]
all_books = romance_books + adventure_books


# frequency of words in a specific list
def word_frequency(list):
    return nltk.FreqDist(brown.words(list))


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
all_books2 = [nltk.word_tokenize(brown.raw()) for book in all_books]

for i in all_books:
    train_data.append([word.words() for word in all_books2[i]
                       if word in all_words])

# creating test and training set

# test print
## print(all_words)
