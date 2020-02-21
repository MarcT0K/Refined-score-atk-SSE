import email
import glob
import multiprocessing
import os

from functools import reduce
from contextlib import contextmanager
from collections import Counter
from typing import List

import nltk
import numpy as np
import pandas as pd
import tqdm

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("stopwords")
nltk.download("punkt")


def get_voc_from_one_email(email_text, freq=False):
    stopwords_list = stopwords.words("english")
    stopwords_list.extend(["subject", "cc", "from", "to", "forward"])
    stemmer = PorterStemmer()

    stemmed_word_list = [
        stemmer.stem(word.lower())
        for sentence in sent_tokenize(email_text)
        for word in word_tokenize(sentence)
        if word.lower() not in stopwords_list and word.isalnum()
    ]
    if freq:
        return nltk.FreqDist(stemmed_word_list)
    else:
        return stemmed_word_list


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class OccRowComputer:
    def __init__(self, voc):
        self.voc = voc

    def __call__(self, word_list):
        return [int(voc_word in word_list) for voc_word in self.voc]


## On peut surement réduire et utiliser qu'une seule fonction
def build_occurence_array(voc: List, freq_dict: dict) -> pd.DataFrame:
    occ_list = []
    with poolcontext(processes=multiprocessing.cpu_count()) as pool:
        for row in tqdm.tqdm(
            pool.imap_unordered(OccRowComputer(voc), freq_dict.values()),
            desc="Computing the occurence array",
            total=len(freq_dict.values()),
        ):
            occ_list.append(row)

    return np.array(occ_list)


def compute_coocc_matrix(occ_array: pd.DataFrame):
    coocc_mat = np.dot(occ_array.T, occ_array)
    np.fill_diagonal(coocc_mat, 0)
    return coocc_mat


def extract_email_voc(ind, df, one_occ_per_doc=True):
    freq_dict = {}
    glob_freq_list = {}
    for row_tuple in tqdm.tqdm(
        iterable=df.itertuples(),
        desc=f"Extracting corpus vocabulary (Core {ind})",
        total=len(df),
        position=ind,
    ):
        temp_freq_dist = get_voc_from_one_email(row_tuple.mail_body, freq=True)
        freq_dict[row_tuple.filename] = []
        for word, freq in temp_freq_dist.items():
            freq_to_add = 1 if one_occ_per_doc else freq
            freq_dict[row_tuple.filename].append(word)
            try:
                glob_freq_list[word] += freq_to_add
            except KeyError:
                glob_freq_list[word] = freq_to_add
    return freq_dict, glob_freq_list


def merge_results(res1, res2):
    merge_results2 = Counter(res1[1]) + Counter(res2[1])

    merge_results1 = res1[0].copy()
    merge_results1.update(res2[0])
    return (merge_results1, merge_results2)


def corpus_to_co_occ_mat(corpus_df, voc_size=100, minimum_freq=1):
    glob_freq_dict = {}
    freq_dict = {}
    # Word tokenization
    NUM_CORES = multiprocessing.cpu_count()
    with poolcontext(processes=NUM_CORES) as pool:
        results = pool.starmap(
            extract_email_voc, enumerate(np.array_split(corpus_df, NUM_CORES))
        )
        freq_dict, glob_freq_dict = reduce(merge_results, results)

    del corpus_df
    print(f"Number of unique words (except the stopwords): {len(glob_freq_dict)}")

    # Creation of the vocabulary
    glob_freq_list = nltk.FreqDist(glob_freq_dict)
    del glob_freq_dict
    glob_freq_list = (
        glob_freq_list.most_common(voc_size)
        if voc_size
        else glob_freq_list.most_common()
    )
    voc = [word for word, count in glob_freq_list if count >= minimum_freq]
    print(f"Vocabulary size: {len(voc)}")
    del glob_freq_list

    # Creation of the co-occurence matrix
    occ_array = build_occurence_array(voc=voc, freq_dict=freq_dict)
    if not occ_array.any():
        raise ValueError("Occurence array is empty")
    del freq_dict
    print("Creating the word-word co-occurence matrix")
    return compute_coocc_matrix(occ_array=occ_array), voc


def split_df(df, frac=0.5):
    first_split = df.sample(frac=0.6, random_state=200)
    second_split = df.drop(first_split.index)

    return first_split, second_split


def get_body_from_email(mail):
    """To get the content from raw email"""
    msg = email.message_from_string(mail)
    parts = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            parts.append(part.get_payload())
    return "".join(parts)


def extract_sent_mails_body(maildir_directory="~/research/maildir/") -> pd.DataFrame:
    # We move in the mail directory
    os.chdir(os.path.expanduser(maildir_directory))

    mails = glob.glob("./*/_sent_mail/*")

    mail_contents = []
    for mailfile_path in tqdm.tqdm(iterable=mails, desc="Reading the emails"):
        with open(mailfile_path, "r") as mailfile:
            raw_mail = mailfile.read()
            mail_contents.append(get_body_from_email(raw_mail))

    return pd.DataFrame(data={"filename": mails, "mail_body": mail_contents})
