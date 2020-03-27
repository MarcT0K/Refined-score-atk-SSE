import multiprocessing
import random

from functools import partial, reduce
from typing import Dict, List, Tuple, Optional

import colorlog
import numpy as np
import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from .common import poolcontext

logger = colorlog.getLogger("Keyword Regression Attack")

class PlainCipherAssigner:
    def __init__(
        self,
        plain_occ_array,
        cipher_occ_array,
        plain_sorted_voc: List[Tuple[str, int]],
        known_queries: Dict[str, str],
        cipher_sorted_voc: Optional[List[Tuple[str, int]]] = None,
    ):
        self.plain_number_docs = plain_occ_array.shape[0]
        self.cipher_number_docs = cipher_occ_array.shape[0]
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

        if not known_queries:
            # TODO We could generate the known queries in this class
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Cipher values must be unique.")

        # Known queries dict => keys: plain, values:cipher
        self._known_queries = known_queries

        self._run_coocc_pca(pca_dimensions=10)
        self.prob_diff_std = self._estimate_std_prob()

        self._known_queries_vec = [
            (
                self.plain_reduced[self.plain_voc_info[plain_word]["vector_ind"], :],
                self.cipher_reduced[self.cipher_voc_info[cipher_word]["vector_ind"], :],
            )
            for plain_word, cipher_word in known_queries.items()
        ]

    def _run_coocc_pca(self, pca_dimensions):
        pca = PCA(n_components=pca_dimensions)
        self.plain_reduced = pca.fit_transform(self.plain_coocc)
        self.cipher_reduced = pca.fit_transform(self.cipher_coocc)

    def _estimate_std_prob(self, not_matching_rate=0.6):
        logger.debug("Preparing training pairs")
        training_pairs = [
            (plain, cipher) for plain, cipher in self._known_queries.items()
        ]

        nb_not_matching = int(
            len(self._known_queries) * not_matching_rate / (1 - not_matching_rate)
        )
        plain_unknown = self.plain_voc_info.keys() - self._known_queries.keys()
        possible_pairs = [
            (plain, cipher)
            for plain in plain_unknown
            for cipher in self._known_queries.values()
        ]
        training_pairs.extend(random.sample(possible_pairs, nb_not_matching // 2))

        cipher_unknown = self.cipher_voc_info.keys() - self._known_queries.values()
        possible_pairs = [
            (plain, cipher)
            for cipher in cipher_unknown
            for plain in self._known_queries.values()
        ]
        training_pairs.extend(random.sample(possible_pairs, nb_not_matching + 1 // 2))

        prob_diffs = [
            (
                self.plain_voc_info[plain]["word_prob"]
                - self.cipher_voc_info[cipher]["word_prob"]
            )
            ** 2
            for plain, cipher in training_pairs
        ]

        return np.std(prob_diffs)

    def _word_pair_to_instance(self, plain, cipher):
        plain_ind = self.plain_voc_info[plain]["vector_ind"]
        cipher_ind = self.cipher_voc_info[cipher]["vector_ind"]

        prob_diff = (
            self.plain_voc_info[plain]["word_prob"]
            - self.cipher_voc_info[cipher]["word_prob"]
        ) / self.prob_diff_std

        cos_sim_diffs = []
        for known_plain_vec, known_cipher_vec in self._known_queries_vec:
            plain_cos = cos_sim([self.plain_reduced[plain_ind, :]], [known_plain_vec])
            cipher_cos = cos_sim(
                [self.cipher_reduced[cipher_ind, :]], [known_cipher_vec]
            )
            cos_sim_diffs.append(plain_cos - cipher_cos)
        return np.append(cos_sim_diffs, prob_diff) ** 2

    def word_pair_to_score(self, plain, cipher):
        """Only use if you want to transform one single pair
        """
        instance = self._word_pair_to_instance(plain, cipher)
        mean_cosine_diff = np.mean(instance[:-1])  # TODO: a trimmed mean?
        return 1 / mean_cosine_diff + 1 / instance[-1]

    def _sub_pred(self, ind, cipher_word_list, k):
        prediction = []
        for cipher_kw in tqdm.tqdm(
            iterable=cipher_word_list,
            desc=f"Core {ind}: Evaluating each plain-cipher pairs",
            position=ind,
        ):
            cipher_vec = self.cipher_reduced[
                self.cipher_voc_info[cipher_kw]["vector_ind"], :
            ]
            cipher_prob = self.cipher_voc_info[cipher_kw]["word_prob"]
            cipher_cos_sim = np.array(
                [
                    cos_sim([cipher_vec], [known_cipher_vec])
                    for _plain_vec, known_cipher_vec in self._known_queries_vec
                ]
            )
            score_list = []
            for plain_kw in self.plain_voc_info.keys():
                plain_prob = self.plain_voc_info[plain_kw]["word_prob"]
                plain_vec = self.plain_reduced[
                    self.plain_voc_info[plain_kw]["vector_ind"], :
                ]
                plain_cos_sim = np.array(
                    [
                        cos_sim([plain_vec], [known_plain_vec])
                        for known_plain_vec, _cipher_vec in self._known_queries_vec
                    ]
                )
                prob_diff = plain_prob - cipher_prob
                cos_sim_diff = plain_cos_sim - cipher_cos_sim
                instance = np.append(cos_sim_diff, prob_diff) ** 2
                score = 1 / np.mean(instance[:-1]) + 1 / instance[-1]
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
