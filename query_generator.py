"""Functions simulating the result harvesting of an attacker"""
import multiprocessing

from typing import List, Tuple

import colorlog
import numpy as np
import scipy.stats as stats
import tqdm

from common import KeywordExtractor, poolcontext

logger = colorlog.getLogger("Keyword Alignment Attack")

class QueryResultExtractor(KeywordExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        x = np.arange(1, len(self.sorted_voc) + 1)
        a = 1.0
        weights = x ** (-a)
        weights /= weights.sum()
        self._bounded_zipf = stats.rv_discrete(name="bounded_zipf", values=(x, weights))

    def _extract_query_result(self, query_ind: int):
        """Returns the documents which contains the keyword.
        Please not that it would be more efficient to return the column instead of
        the list of the documents containing the keyword. However, to stick to the attack,
        we don't want to add the number of documents as a background knowledge.

        We could have keep it and not use it but it is more rigorous this way.
        
        Arguments:
            query_ind {int} -- Cipher keyword index
        
        Returns:
            Tuple[int, List[int]] -- (keyword_ind, [document_ids])
        """
        query_column = self.occ_array[:, query_ind]
        return (
            self.sorted_voc[query_ind][0],
            [doc_ind for doc_ind, val in enumerate(query_column) if val],
        )

    def get_query_answer(self, size=1) -> dict:
        """Returns ONLY unique queries. Duplicate queries are removed from the result.
        Uses a zipfian law to generate the queries then there are several duplicates.
        
        Keyword Arguments:
            size {int} -- Size of the sample of random queries (default: {1})
        
        Returns:
            dict -- trapdoor_id: [document_ids]
            maximum size is the size of the random sample
        """
        sample = self._bounded_zipf.rvs(size=size) - 1
        unique_sample = set(sample)
        logger.debug(f"Number of duplicate queries: {len(sample) - len(unique_sample)}")
        query_answer_dict = {}
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            for couple in tqdm.tqdm(
                pool.imap_unordered(self._extract_query_result, unique_sample),
                desc="Generating query-answer couples",
                total=len(unique_sample),
            ):
                query_answer_dict[couple[0]] = couple[1]
        return query_answer_dict