import multiprocessing
import random

from functools import partial, reduce
from typing import Dict, List, Tuple, Optional

import colorlog
import numpy as np
import tqdm

from .common import poolcontext

logger = colorlog.getLogger("Keyword Regression Attack")


class PlainCipherAssigner:
    _norm = np.max  #  It is the Chebyshev norm (also called infinity or uniform norm

    def __init__(
        self,
        plain_occ_array,
        cipher_occ_array,
        plain_sorted_voc: List[Tuple[str, int]],
        known_queries: Dict[str, str],
        cipher_sorted_voc: Optional[List[Tuple[str, int]]] = None,
    ):
        if not known_queries:
            # TODO We could generate the known queries in this class
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Cipher values must be unique.")

        self._known_queries = known_queries

        self.plain_number_docs = plain_occ_array.shape[0]
        self.cipher_number_docs = self.__estimate_cipher_size(
            dict(plain_sorted_voc), dict(cipher_sorted_voc)
        )
        # Background knowledge
        self.plain_voc_info = {
            word: {"vector_ind": ind, "word_prob": occ / self.plain_number_docs}
            for ind, (word, occ) in enumerate(plain_sorted_voc)
        }
        self.cipher_voc_info = (
            {  # "word" can be either the plain word or the trapdoor
                word: {"vector_ind": ind, "word_prob": occ / self.cipher_number_docs}
                for ind, (word, occ) in enumerate(cipher_sorted_voc)
            }
            if cipher_sorted_voc is not None
            else {}
        )

        logger.info("Computing cooccurrence matrices")
        self.plain_coocc = (
            np.dot(plain_occ_array.T, plain_occ_array) / self.plain_number_docs
        )
        np.fill_diagonal(self.plain_coocc, 0)

        self.cipher_coocc = (
            np.dot(cipher_occ_array.T, cipher_occ_array) / self.cipher_number_docs
        )
        np.fill_diagonal(self.cipher_coocc, 0)

    def __estimate_cipher_size(self, plain_voc, cipher_voc):
        """Estimates the number of documents stored.
        """
        nb_doc_ratio_estimator = np.mean(
            [
                cipher_voc[cipher] / plain_voc[plain]
                for plain, cipher in self._known_queries.items()
            ]
        )

        return self.plain_number_docs * nb_doc_ratio_estimator

    def set_norm(self, norm_function):
        self._norm = norm_function

    def _sub_pred(self, ind, cipher_word_list, k):
        prediction = []
        for cipher_kw in tqdm.tqdm(
            iterable=cipher_word_list,
            desc=f"Core {ind}: Evaluating each plain-cipher pairs",
            position=ind,
        ):
            cipher_ind = self.cipher_voc_info[cipher_kw]["vector_ind"]
            cipher_coocc = np.array(
                [
                    self.cipher_coocc[
                        cipher_ind, self.cipher_voc_info[known_cipher]["vector_ind"]
                    ]
                    for known_cipher in self._known_queries.values()
                ]
            )
            score_list = []
            for plain_kw in self.plain_voc_info.keys():
                plain_ind = self.plain_voc_info[plain_kw]["vector_ind"]
                plain_coocc = np.array(
                    [
                        self.plain_coocc[
                            plain_ind, self.plain_voc_info[known_plain]["vector_ind"]
                        ] #  Access cost can be optimized by storing known indices lists
                        for known_plain in self._known_queries.keys()
                    ]
                )
                cocc_diff = (plain_coocc - cipher_coocc) ** 2
                norm = self._norm(cocc_diff)
                if norm:
                    score = -np.log(norm)
                else:
                    score = np.inf  # perfect match
                score_list.append((score, plain_kw))
            score_list.sort(key=lambda tup: tup[0])
            best_candidates = [word for _score, word in score_list[-k:]]
            prediction.append((cipher_kw, best_candidates))
        return prediction

    def predict(self, cipher_word_list, k):
        prediction = {}
        NUM_CORES = multiprocessing.cpu_count()
        with poolcontext(processes=NUM_CORES) as pool:
            pred_func = partial(self._sub_pred, k=k)
            results = pool.starmap(
                pred_func,
                enumerate([cipher_word_list[i::NUM_CORES] for i in range(NUM_CORES)]),
            )
            prediction = dict(reduce(lambda x, y: x + y, results))
        return prediction

    def accuracy(self, k=3, eval_dico=None):
        # TODO: utiliser le target-weapon assignement pour avoir une meilleure attribution
        # On pourra rafiner en se basant sur l'inertie par rapport aux scores
        assert k > 0
        assert self.cipher_voc_info

        if eval_dico is None:
            logger.warning("Experimental setup is enabled.")
            # Cipher vocabulary has not truly been ciphered.
            # It makes the accuracy easier to compute since we just
            # need to compare the guess with the corresponding keyword
            local_eval_dico = {kw: kw for kw in self.cipher_voc_info.keys()}
        else:
            local_eval_dico = eval_dico

        eval_words = set(self.cipher_voc_info.keys()).intersection(
            local_eval_dico.keys()
        )
        res_dict = self.predict(list(eval_words), k)  # keys: cipher, values: plain

        eval_res = [local_eval_dico[word] in res_dict[word] for word in eval_words]
        accuracy = sum(eval_res) / len(eval_res)
        return accuracy, res_dict
