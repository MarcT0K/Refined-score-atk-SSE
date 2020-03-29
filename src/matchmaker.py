import multiprocessing
import random

from functools import partial, reduce
from typing import Dict, List, Tuple, Optional

import colorlog
import numpy as np
import tqdm

from .common import poolcontext

logger = colorlog.getLogger("Keyword Regression Attack")


class KeywordTrapdoorMatchmaker:
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc: List[Tuple[str, int]],
        known_queries: Dict[str, str],
        trapdoor_sorted_voc: Optional[List[Tuple[str, int]]],
        norm_ord=2,  # L2 (Euclidean norm)
    ):
        self.set_norm_ord(norm_ord=norm_ord)

        if not known_queries:
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Several trapdoors are linked to the same keyword.")

        self._known_queries = known_queries  # Keys: trapdoor, Values: keyword

        self.number_similar_docs = keyword_occ_array.shape[0]
        self.number_real_docs = self.__estimate_nb_real_docs(
            dict(keyword_sorted_voc), dict(trapdoor_sorted_voc)
        )

        # NB: kw=KeyWord; td=TrapDoor
        self.kw_voc_info = {
            word: {"vector_ind": ind, "word_occ": occ / self.number_similar_docs}
            for ind, (word, occ) in enumerate(keyword_sorted_voc)
        }

        self.td_voc_info = {
            word: {"vector_ind": ind, "word_occ": occ / self.number_real_docs}
            for ind, (word, occ) in enumerate(trapdoor_sorted_voc)
        }

        logger.info("Computing cooccurrence matrices")
        kw_coocc = (
            np.dot(keyword_occ_array.T, keyword_occ_array) / self.number_similar_docs
        )
        np.fill_diagonal(kw_coocc, 0)
        ind_known_kw = [
            self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()
        ]
        self.kw_reduced_coocc = kw_coocc[:, ind_known_kw]

        td_coocc = (
            np.dot(trapdoor_occ_array.T, trapdoor_occ_array) / self.number_real_docs
        )
        np.fill_diagonal(td_coocc, 0)
        ind_known_td = [
            self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()
        ]
        self.td_reduced_coocc = td_coocc[:, ind_known_td]

    def __estimate_nb_real_docs(self, kw_voc, td_voc):
        """Estimates the number of documents stored.
        """
        nb_doc_ratio_estimator = np.mean(
            [td_voc[td] / kw_voc[kw] for td, kw in self._known_queries.items()]
        )

        return self.number_similar_docs * nb_doc_ratio_estimator

    def set_norm_ord(self, norm_ord):
        self._norm = partial(np.linalg.norm, ord=norm_ord)

    def _sub_pred(self, ind, td_list, k):
        prediction = []
        for trapdoor in tqdm.tqdm(
            iterable=td_list,
            desc=f"Core {ind}: Evaluating each plain-cipher pairs",
            position=ind,
        ):
            try:
                trapdoor_ind = self.td_voc_info[trapdoor]["vector_ind"]
            except KeyError:
                logger.debug(f"Unknown trapdoor: {trapdoor}")
                prediction.append((trapdoor, []))
                continue
            trapdoor_vec = self.td_reduced_coocc[trapdoor_ind]

            score_list = []
            for keyword, kw_info in self.kw_voc_info.items():
                keyword_vec = self.kw_reduced_coocc[kw_info["vector_ind"]]
                vec_diff = keyword_vec - trapdoor_vec
                td_kw_distance = self._norm(vec_diff)
                if td_kw_distance:
                    score = -np.log(td_kw_distance)
                else:
                    score = np.inf
                score_list.append((keyword, score))
            score_list.sort(key=lambda tup: tup[1])
            best_candidates = [kw for kw, _score in score_list[-k:]]
            prediction.append((trapdoor, best_candidates))
        return prediction

    def predict(self, trapdoor_list, k):
        prediction = {}
        NUM_CORES = multiprocessing.cpu_count()
        with poolcontext(processes=NUM_CORES) as pool:
            pred_func = partial(self._sub_pred, k=k)
            results = pool.starmap(
                pred_func,
                enumerate([trapdoor_list[i::NUM_CORES] for i in range(NUM_CORES)]),
            )
            prediction = dict(reduce(lambda x, y: x + y, results))
        return prediction

    def accuracy(self, k=1, eval_dico=None):
        assert k > 0
        assert self.td_voc_info

        if eval_dico is None:
            logger.warning("Experimental setup is enabled.")
            # Trapdoor vocabulary is not a list of tokens but directly the list of keywords.
            local_eval_dico = {kw: kw for kw in self.td_voc_info.keys()}
        else:
            local_eval_dico = eval_dico
            # Keys: Trapdoors; Values: Keywords
        eval_trapdoors = list(local_eval_dico.keys())

        res_dict = self.predict(trapdoor_list=eval_trapdoors, k=k)
        match_list = [
            eval_kw in res_dict[eval_td] for eval_td, eval_kw in local_eval_dico.items()
        ]
        recovery_rate = sum(match_list) / len(local_eval_dico)
        return recovery_rate, match_list
