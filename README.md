# Extracting Important Words from An Email Using TF-IDF

This repository accompanies my [YouTube tutorial](https://youtu.be/KYLscCskTtw) on text mining with tf–idf in Python.

tf–idf is a common technique for assigning importance to words in a document. It is based on the term's frequency in the document (tf) and in a baseline corpus (idf).

# Installation

This repository uses [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html) to manage the environment and dependencies.

Install Pipenv. Once you have it,

`$ pipenv install`

to get all the dependencies.

## preprocess.py

This script generates an `idf` dictionary for the [corpus of Hillary Clinton's emails](https://www.kaggle.com/kaggle/hillary-clinton-emails), obtained from Kaggle. It then saves it as a pickle file, to be consumed by `get_tfidfs.py`.


`$ python preprocess.py`

## get_tfidfs.py

Takes a text file with an input and generates a dictionary of the tokens in the email, along with their corresponding tf–idf scores, based on the Clinton email corpus.

`$ python get_idfs.py sample_emails/aws.txt

