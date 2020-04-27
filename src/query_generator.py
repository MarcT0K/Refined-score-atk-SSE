"""Functions simulating the result harvesting of an attacker"""
import math
import random

from typing import List, Tuple

import colorlog
import numpy as np
import scipy.stats as stats
import tqdm

from .common import KeywordExtractor

logger = colorlog.getLogger("Keyword Regression Attack")


class QueryResultExtractor(KeywordExtractor):
    """Just a keyword extractor augmented with a query generator.
    It corresponds to the index in the server. The fake queries are
    seen by the attacker.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        x = np.arange(1, len(self.sorted_voc_with_occ) + 1)
        a = 1.0
        weights = x ** (-a)
        weights /= weights.sum()
        self._bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))

    def _generate_fake_queries(self, size=1) -> dict:
        """Uses a zipfian law to generate unique queries then there are several duplicates.
        We keep generating queries until having a queryset with the expected size.
        
        Keyword Arguments:
            size {int} -- Size of the sample of random queries (default: {1}) 

        Returns:
            queries_array, sorted_voc -- i-th column corresponds to the i-th word of the sorted voc
        """
        logger.info("Generating fake queries")
        sample_set = set(self._bounded_zipf.rvs(size=size) - 1)
        queries_remaining = size - len(sample_set)
        while queries_remaining > 0:
            # The process is repeated until having the correct size.
            # It is not elegant, but in IKK and Cash, they present
            # queryset with unique queries.
            sample_set = sample_set.union(
                self._bounded_zipf.rvs(size=queries_remaining) - 1
            )
            queries_remaining = size - len(sample_set)
        sample_list = list(sample_set)  # Cast needed to index np.array
        # sample_list is sorted since a set of integers is sorted
        query_voc = [self.sorted_voc_with_occ[ind][0] for ind in sample_list]
        query_columns = self.occ_array[:, sample_list]
        # i-th element of column is 0/1 depending if the i-th document includes the keyword
        return query_columns, query_voc

    def get_fake_queries(self, size=1, hide_nb_files=True) -> dict:
        query_arr, query_voc = self._generate_fake_queries(size=size)

        if hide_nb_files:
            # We remove every line containing only zeros, so we hide the nb of documents stored
            # i.e. we remove every documents not involved in the queries
            query_arr = query_arr[~np.all(query_arr == 0, axis=1)]
        return query_arr, query_voc


class ObfuscatedResultExtractor(QueryResultExtractor):
    def __init__(self, *args, m=6, p=0.88703, q=0.04416, **kwargs):
        self.occ_array = np.array([])  # useless but pylint is happy now :)
        super().__init__(*args, **kwargs)
        self._p = p
        self._q = q
        self._m = m

        nrow, ncol = self.occ_array.shape
        self.occ_array = np.repeat(self.occ_array, self._m, axis=0)

        for i in range(nrow):
            for j in range(ncol):
                if self.occ_array[i, j]:  # Document i contains keyword j
                    if random.random() < self._p:
                        self.occ_array[i, j] = 0
                else:
                    if random.random() < self._q:
                        self.occ_array[i, j] = 1


class PaddedResultExtractor(QueryResultExtractor):
    def __init__(self, *args, n=100, **kwargs):
        self.occ_array = np.array([])
        super().__init__(*args, **kwargs)
        self._n = n

        _, ncol = self.occ_array.shape
        self._number_real_entries = np.sum(self.occ_array)
        for j in range(ncol):
            nb_entries = sum(self.occ_array[:, j])
            nb_fake_entries_to_add = int(
                math.ceil(nb_entries / self._n) * self._n - nb_entries
            )
            possible_fake_entries = list(np.argwhere(self.occ_array[:, j] == 0).flatten())
            if len(possible_fake_entries) < nb_fake_entries_to_add:
                # We need more documents to generate enough fake entries
                # So we generate fake document IDs
                fake_documents = np.zeros(
                    (nb_fake_entries_to_add - len(possible_fake_entries), ncol)
                )
                self.occ_array = np.concatenate((self.occ_array, fake_documents))
                possible_fake_entries = list(
                    np.argwhere(self.occ_array[:, j] == 0).flatten()
                )
            fake_entries = random.sample(possible_fake_entries, nb_fake_entries_to_add)
            self.occ_array[fake_entries, j] = 1

        self._number_observed_entries = np.sum(self.occ_array)
        logger.debug(
            f"Padding overhead: {self._number_observed_entries/self._number_real_entries}"
        )