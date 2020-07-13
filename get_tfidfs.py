"""
python get_tfidfs.py [filename]

returns the input text split into tokens, with tf-idf scores

{
    "elephant": 45.1,
    "chair": 13,
    ...
}
"""
import argparse
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter


def get_tfidfs(input_email):
    """Get the list of terms and corresponding
    tf-idf scores"""
    with open("idf_results.p", "rb") as idf_pickle:
        idfs = pickle.load(idf_pickle)
    tokens = word_tokenize(input_email.lower())
    tfs = Counter(tokens)
    tfidfs = {}
    for term in tfs:
        if term in idfs:
            tfidfs[term] = tfs[term] * idfs[term]
    counter = Counter(tfidfs).most_common()
    return dict(counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_msg = "File path to input email"
    parser.add_argument("filename", help=help_msg)
    args = parser.parse_args()
    with open(args.filename) as file:
        input_email = file.read()
    tfidfs = get_tfidfs(input_email)
    print(tfidfs)
