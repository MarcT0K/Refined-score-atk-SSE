"""Functions simulating the result harvesting of an attacker"""
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
        super().__init__(*args, **kwargs)
        self._p = p
        self._q = q
        self._m = m

    def _generate_fake_queries(self, size=1) -> dict:
        query_arr, query_voc = super()._generate_fake_queries(size=size)
        nrow, ncol = query_arr.shape
        query_arr = np.repeat(query_arr, self._m, axis=0)

        for i in range(nrow):
            for j in range(ncol):
                if query_arr[i, j]:  # Document i contains keyword j
                    if random.random() < self._p:
                        query_arr[i, j] = 0
                else:
                    if random.random() < self._q:
                        query_arr[i, j] = 1
        return query_arr, query_voc
