"""Functions simulating the result harvesting of an attacker"""
from typing import List, Tuple

import colorlog
import numpy as np
import scipy.stats as stats
import tqdm

from .common import KeywordExtractor

logger = colorlog.getLogger("Keyword Regression Attack")


class QueryResultExtractor(KeywordExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        x = np.arange(1, len(self.sorted_voc) + 1)
        a = 1.0
        weights = x ** (-a)
        weights /= weights.sum()
        self._bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))

    def get_fake_queries(self, size=1) -> dict:
        """Uses a zipfian law to generate unique queries then there are several duplicates.
        We keep generating queries until having a queryset with the expected size.
        
        Keyword Arguments:
            size {int} -- Size of the sample of random queries (default: {1})
        
        Returns:
            queries_array, sorted_voc
        """
        logger.info("Generating fake queries")
        sample_set = set(self._bounded_zipf.rvs(size=size) - 1)
        queries_remaining = size - len(sample_set)
        while queries_remaining > 0:
            sample_set = sample_set.union(
                self._bounded_zipf.rvs(size=queries_remaining) - 1
            )
            queries_remaining = size - len(sample_set)
        sample_list = list(sample_set)  # Cast needed to index np.array
        query_voc = [self.sorted_voc[ind] for ind in sample_list]

        return self.occ_array[:, sample_list], query_voc
