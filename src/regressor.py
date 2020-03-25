import random

from typing import Dict, List, Tuple, Optional

import colorlog
import numpy as np
import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


logger = colorlog.getLogger("Keyword Regression Attack")


class PlainCipherRegressor:
    def __init__(
        self,
        plain_occ_array,
        cipher_occ_array,
        plain_sorted_voc: List[Tuple[str, int]],
        known_queries: Dict[str, str],
        cipher_sorted_voc: Optional[List[Tuple[str, int]]] = None,
        **kwargs
    ):
        self._reg = LogisticRegression(**kwargs)

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

        known_plain_for_comp = random.sample(
            known_queries.keys(), int(0.25 * len(known_queries))
        )
        # Known queries dict => keys: plain, values:cipher
        self._known_ind_for_comp = [
            (
                self.plain_voc_info[plain_word]["vector_ind"],
                self.cipher_voc_info[known_queries[plain_word]]["vector_ind"],
            )
            for plain_word in known_plain_for_comp
        ]
        # self._known_queries_for_training = {
        #     plain: cipher
        #     for plain, cipher in known_queries
        #     if plain not in known_plain_for_comp
        # } TODO: see if we need to extract the comparison points
        self._known_queries_for_training = known_queries

    def _run_pca(self, pca_dimensions):
        pca = PCA(n_components=pca_dimensions)
        self.plain_reduced = pca.fit_transform(self.plain_coocc)
        self.cipher_reduced = pca.fit_transform(self.cipher_coocc)

    def _generate_training_data(self, pca_dimensions=20, not_matching_rate=0.6):
        # normalisation bien faite?
        # Voir la variation des proba entre deux échantillons => c'est la derniere col!!
        self._run_pca(pca_dimensions)

        X_train = np.empty((0, len(self._known_ind_for_comp) + 1), dtype=float)
        y_train = np.array([], dtype=int)

        logger.debug("Preparing training pairs")
        training_pairs = [
            (plain, cipher, 1)
            for plain, cipher in self._known_queries_for_training.items()
        ]

        nb_not_matching = int(
            len(self._known_queries_for_training)
            * not_matching_rate
            / (1 - not_matching_rate)
        )
        plain_unknown = (
            self.plain_voc_info.keys() - self._known_queries_for_training.keys()
        )
        possible_pairs = [
            (plain, cipher, 0)
            for plain in plain_unknown
            for cipher in self._known_queries_for_training.values()
        ]
        training_pairs.extend(random.sample(possible_pairs, nb_not_matching // 2))

        cipher_unknown = (
            self.cipher_voc_info.keys() - self._known_queries_for_training.values()
        )
        possible_pairs = [
            (plain, cipher, 0)
            for cipher in cipher_unknown
            for plain in self._known_queries_for_training.values()
        ]
        training_pairs.extend(random.sample(possible_pairs, nb_not_matching + 1 // 2))

        for plain, cipher, res in tqdm.tqdm(
            iterable=training_pairs, desc="Generating instances"
        ):
            instance = self._word_pair_to_instance(plain, cipher)
            X_train = np.append(X_train, [instance], axis=0)
            y_train = np.append(y_train, res)

        return X_train, y_train

    def _word_pair_to_instance(self, plain, cipher):
        plain_ind = self.plain_voc_info[plain]["vector_ind"]
        cipher_ind = self.cipher_voc_info[cipher]["vector_ind"]

        prob_diff = (
            self.plain_voc_info[plain]["word_prob"]
            - self.cipher_voc_info[cipher]["word_prob"]
        )

        cos_sim_diffs = []
        for known_plain_ind, known_cipher_ind in self._known_ind_for_comp:
            # TODO: See if we should just compare from a part of the known queries
            plain_cos = cos_sim(
                [self.plain_reduced[plain_ind, :]],
                [self.plain_reduced[known_plain_ind, :]],
            )
            cipher_cos = cos_sim(
                [self.cipher_reduced[cipher_ind, :]],
                [self.cipher_reduced[known_cipher_ind, :]],
            )
            cos_sim_diffs.append(plain_cos - cipher_cos)
        return np.append(cos_sim_diffs, prob_diff) ** 2

    def fit(self, *args, **kwargs):
        X_train, y_train = self._generate_training_data(*args, **kwargs)
        logger.debug("Fitting the logistic regression")
        self._reg.fit(X_train, y_train)  # TODO: see if we should set sample weights

    def get_params(self, deep=True):
        return self._reg.get_params(deep)

    def predict_proba(self, plain, cipher):  # TODO: multiproc + see for the lock
        instance = None

    def accuracy(self, k=3, eval_dico=None):
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

        for cipher_kw in self.cipher_voc_info.keys():
            pass
