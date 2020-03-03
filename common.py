import logging
import multiprocessing
import os
import subprocess

from functools import reduce
from contextlib import contextmanager
from collections import Counter
from typing import List

import colorlog
import nltk
import numpy as np
import pandas as pd
import tqdm

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from glove_file_management import build_voc_file, build_cooccurence_bin

nltk.download("stopwords")
nltk.download("punkt")

logger = colorlog.getLogger("Keyword Alignment Attack")


def setup_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s %(levelname)s]%(reset)s %(module)s: "
            "%(white)s%(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


class OccRowComputer:
    def __init__(self, sorted_voc):
        self.voc = [word for word, occ in sorted_voc]

    def __call__(self, word_list):
        return [int(voc_word in word_list) for voc_word in self.voc]


class KeywordExtractor:
    def __init__(self, corpus_df, voc_size=100, min_freq=1):
        glob_freq_dict = {}
        freq_dict = {}
        # Word tokenization
        NUM_CORES = multiprocessing.cpu_count()
        with poolcontext(processes=NUM_CORES) as pool:
            results = pool.starmap(
                self.extract_email_voc, enumerate(np.array_split(corpus_df, NUM_CORES))
            )
            freq_dict, glob_freq_dict = reduce(self._merge_results, results)

        del corpus_df
        logger.info(
            f"Number of unique words (except the stopwords): {len(glob_freq_dict)}"
        )

        # Creation of the vocabulary
        glob_freq_list = nltk.FreqDist(glob_freq_dict)
        del glob_freq_dict
        glob_freq_list = (
            glob_freq_list.most_common(voc_size)
            if voc_size
            else glob_freq_list.most_common()
        )
        self.sorted_voc = sorted(
            [(word, count) for word, count in glob_freq_list if count >= min_freq],
            key=lambda d: d[1],
            reverse=True,
        )
        logger.info(f"Vocabulary size: {len(self.sorted_voc)}")
        del glob_freq_list

        # Creation of the co-occurence matrix
        self.occ_array = self.build_occurence_array(
            sorted_voc=self.sorted_voc, freq_dict=freq_dict
        )
        if not self.occ_array.any():
            raise ValueError("Occurence array is empty")

    @staticmethod
    def _merge_results(res1, res2):
        merge_results2 = Counter(res1[1]) + Counter(res2[1])

        merge_results1 = res1[0].copy()
        merge_results1.update(res2[0])
        return (merge_results1, merge_results2)

    @staticmethod
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

    @staticmethod
    def extract_email_voc(ind, df, one_occ_per_doc=True):
        freq_dict = {}
        glob_freq_list = {}
        for row_tuple in tqdm.tqdm(
            iterable=df.itertuples(),
            desc=f"Extracting corpus vocabulary (Core {ind})",
            total=len(df),
            position=ind,
        ):
            temp_freq_dist = KeywordExtractor.get_voc_from_one_email(
                row_tuple.mail_body, freq=True
            )
            freq_dict[row_tuple.filename] = []
            for word, freq in temp_freq_dist.items():
                freq_to_add = 1 if one_occ_per_doc else freq
                freq_dict[row_tuple.filename].append(word)
                try:
                    glob_freq_list[word] += freq_to_add
                except KeyError:
                    glob_freq_list[word] = freq_to_add
        return freq_dict, glob_freq_list

    @staticmethod
    def build_occurence_array(sorted_voc: List, freq_dict: dict) -> pd.DataFrame:
        occ_list = []
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            for row in tqdm.tqdm(
                pool.imap_unordered(OccRowComputer(sorted_voc), freq_dict.values()),
                desc="Computing the occurence array",
                total=len(freq_dict.values()),
            ):
                occ_list.append(row)

        return np.array(occ_list, dtype=np.float64)


class CooccurenceBuilder:
    def __init__(
        self,
        vocab_filename: str = "vocab.txt",
        vector_filename: str = "vectors.txt",
        voc_with_occ=None,
        occ_array=None,
        query_ans_dict=None,
    ):
        self.vocab_filename = vocab_filename
        self.vector_filename = vector_filename

        if (
            (occ_array is not None or voc_with_occ is not None)
            == query_ans_dict
            is not None
        ):
            raise ValueError("You must provide either a list of tuples or array+voc")
        if (occ_array is not None) != (voc_with_occ is not None):
            raise ValueError("Occurence array AND vocabulary are needed")

        if voc_with_occ:
            self.sorted_voc = voc_with_occ
        else:
            voc = {
                voc_cipher: len(doc_ids)
                for voc_cipher, doc_ids in query_ans_dict.items()
            }
            self.sorted_voc = sorted(voc.items(), key=lambda d: d[1], reverse=True)
            # To know, which word corresponds to the j-th column
            # You just need to look at the sorted voc and pick j-th element

            max_doc_id = reduce(
                lambda max_int, list_with_int: max([max_int] + list_with_int),
                [0] + list(query_ans_dict.values()),
            )
            occ_array = np.zeros((max_doc_id + 1, len(self.sorted_voc)))

            for kw_ind, (cipher, _occ) in enumerate(self.sorted_voc):
                for doc_id in query_ans_dict[cipher]:
                    occ_array[doc_id, kw_ind] = 1

        self.coocc_mat = np.dot(occ_array.T, occ_array).astype(int)
        np.fill_diagonal(self.coocc_mat, 0)

    def get_voc_dict(self):
        return {word: occ for word, occ in self.sorted_voc}

    def generate_glove_files(self):
        build_voc_file(self.sorted_voc)
        build_cooccurence_bin(self.coocc_mat)

    def launch_glove(
        self,
        glove_dir="~/research/code/GloVe",
        vector_size=50,
        iter_nb=30,
        mem_available=1.0,
    ):
        path = os.path.expanduser(glove_dir)
        os.system(
            f"{path}/build/shuffle -memory {mem_available} -verbose 2 "
            f"< cooccurrence.bin > cooccurrence.shuf.bin"
        )
        subprocess.run(
            [
                f"{path}/build/glove",
                "-save-file",
                self.vector_filename,
                "-threads",
                "8",
                "-input-file",
                "cooccurrence.shuf.bin",
                "-x-max",
                "10",
                "-iter",
                f"{iter_nb}",
                "-vector-size",
                f"{vector_size}",
                "-binary",
                "2",
                "-vocab-file",
                self.vocab_filename,
                "-verbose",
                "2",
            ]
        )

