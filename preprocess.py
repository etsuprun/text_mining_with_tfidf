import math
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation


def get_terms(email):
    """1) Lowercase the email
    2) Tokenize the email into words
    3) Remove stop words and punctuation
    """
    tokens = word_tokenize(email.lower())
    words_to_remove = stopwords.words("english")
    words_to_remove.extend(list(punctuation))
    return [t for t in set(tokens) if not t in words_to_remove]


def get_term_doc(emails):
    """Turn tokenized emails into a term-document dictionary"""
    term_doc = {}
    for i, email in enumerate(emails):
        for term in email:
            if term in term_doc:
                term_doc[term].add(i)
            else:
                term_doc[term] = {i}
    return term_doc


def compute_idfs(term_doc, corpus_size):
    """
    Calculate idfs for all the terms
    idf = log2 (corpus size / doc freq)
    """
    idfs = {}
    for term in term_doc:
        doc_freq = len(term_doc[term])
        idfs[term] = math.log2(corpus_size / doc_freq)
    return idfs


if __name__ == "__main__":
    # reading in the E-mails from the Hillary Clinton email corpus
    emails = (pd.read_csv("Emails.csv")
              ["ExtractedBodyText"]
              .dropna()
              .tolist())
    tokenized_emails = [get_terms(email) for email in emails]
    term_doc = get_term_doc(tokenized_emails)
    idfs = compute_idfs(term_doc, len(emails))

    # we will use this pickle in the get_tfidfs.py script
    with open("idf_results.p", "wb") as idf_file:
        pickle.dump(idfs, idf_file)
