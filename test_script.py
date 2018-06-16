# this script is to test theories separate in
#    an environment separate from the genre_classifier

import nltk
import random
from nltk.corpus import brown

# stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
custom_stopwords = set(('romance', 'adventure', 'page', 'chapter', 'said'))
all_stopwords = default_stopwords | custom_stopwords

# genres to be classified
#    need to allow for all genres
# tasks: allow for classifier to accept txt files
categories = ['romance', 'adventure']

# tokenize words per book per genre in list [(book1_words, genre),...]
books = list()
for genre in categories:
        books.extend(
                (list(nltk.word_tokenize(open(brown.abspath(book_file)).read())
                      ), genre) for book_file in brown.fileids(str(genre)))

# len of stored books for training/test division
books_loaded = len(books)

# randomize loaded books 
random.shuffle(books)

# Extract features as top 2000 most occuring words in books
def book_features(book):
    # Find 2000 most frequent words in book
    most_frequent = nltk.FreqDist(word.partition('/')[0].lower() for word in book
                                  if not word.partition('/')[0].isdigit() and  # Remove numbers
                                  word.partition('/')[0].isalnum() and  # Remove punctuation
                                  len(word.partition('/')[0]) > 1 and  # Remove single characters
                                  word.partition('/')[0] not in all_stopwords  # Remove stopwords
                                  )
    most_frequent = list(most_frequent)[:2000]  # Only top 2000

    # Define unique word occurances in book
    book_words = set(book)
    features = {}

    # For each unique word in the book found in most frequent words in the book
    #   add the 'word' to a dictionary with corresponding key, contains('word')
    for word in most_frequent:
        features['contains({})'.format(word)] = (word in book_words)
    return features


# Seperate data into training and test sets and save data for later
#     research methods to improve upon testing
featuresets = [(book_features(book), genre) for (book, genre) in books]
test_set = featuresets[:int(books_loaded * .35)]  # Reserve 35 percent for tests
train_set = featuresets[int(books_loaded * .35):]  # Remaining is training

# Training Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# incorrect approach
# # training and test data (words per book per genre)
# train_data = [brown.words(fileid) for fileid in brown.fileids('romance')][:19]
# test_data = [brown.words(fileid) for fileid in brown.fileids('romance')][19:]

# # combining training and test lists simultaneously (different lists)
# #    words in genre (need words per book per genre)
# train_freq = list()
# for word in range(len(train_data)):
#     train_freq.extend(train_data[word])

# test_freq = list()
# for word in range(len(test_data)):
#     test_freq.extend(test_data[word])

# # so this takes each individual char from the list
# #    and combines them
# # redundant - the list was already combined above
# # tb_clean = [word for word in train_freq for word in word]

# # cleaned training and test data
# clean_train = list()
# clean_train.extend([words.lower() for words in train_freq
#                    if not words.isnumeric() and  # Remove numbers
#                    len(words) > 1 and  # Remove single characters
#                    words not in all_stopwords])  # Remove stopwords

# clean_test = list()
# clean_test.extend([words.lower() for words in test_freq
#                   if not words.isnumeric() and  # Remove numbers
#                   len(words) > 1 and  # Remove single characters
#                   words not in all_stopwords])  # Remove stopwords
