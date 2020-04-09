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
        """Initialization of the matchmaker
        
        Arguments:
            keyword_occ_array {np.array} -- Keyword occurrence (row: similar documents; columns: keywords)
            trapdoor_occ_array {np.array} -- Trapdoor occurrence (row: stored documents; columns: trapdoors)
                                            the documents are unknown (just the identifier has
                                            been seen by the attacker)
            keyword_sorted_voc {List[Tuple[str, int]]} -- Keywoord vocabulary extracted from similar documents.
            known_queries {Dict[str, str]} -- Queries known by the attacker
            trapdoor_sorted_voc {Optional[List[Tuple[str, int]]]} -- The trapdoor voc can be a sorted list of (hash, occ)
                                                                    to hide the underlying keywords.
        
        Keyword Arguments:
            norm_ord {int} -- Order of the norm used by the matchmaker (default: {2})
        """
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
        self.kw_coocc = (
            np.dot(keyword_occ_array.T, keyword_occ_array) / self.number_similar_docs
        )
        np.fill_diagonal(self.kw_coocc, 0)

        self.td_coocc = (
            np.dot(trapdoor_occ_array.T, trapdoor_occ_array) / self.number_real_docs
        )
        np.fill_diagonal(self.td_coocc, 0)
        self.__refresh_reduced_coocc()

    def __refresh_reduced_coocc(self):
        """Refresh the co-occurence matrix based on the known queries.
        """
        ind_known_kw = [
            self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()
        ]
        self.kw_reduced_coocc = self.kw_coocc[:, ind_known_kw]
        ind_known_td = [
            self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()
        ]
        self.td_reduced_coocc = self.td_coocc[:, ind_known_td]

    def __estimate_nb_real_docs(self, kw_voc, td_voc):
        """Estimates the number of documents stored.
        """
        nb_doc_ratio_estimator = np.mean(
            [td_voc[td] / kw_voc[kw] for td, kw in self._known_queries.items()]
        )

        return self.number_similar_docs * nb_doc_ratio_estimator

    def set_norm_ord(self, norm_ord):
        """Set the order of the norm used to compute the scores.
        
        Arguments:
            norm_ord {int} -- norm order
        """
        self._norm = partial(np.linalg.norm, ord=norm_ord)

    def __scores_to_cluster(
        self,
        sorted_scores,
        cluster_max_size=10,
        cluster_min_sensitivity=0.0,
        include_cluster_sep=False,
        include_score=False,
    ):
        """From a list of scores, extracts the highest-score cluster using
        simple-linkage clustering.
        
        Arguments:
            sorted_scores {List[float]} -- Sorted list of the scores
        
        Keyword Arguments:
            cluster_max_size {int} -- maximum size of the prediction clusters (default: {10})
            cluster_min_sensitivity {float} -- minimum leap size. Otherwise returns the
                                                maximum-size cluster (default: {0.0})
            include_cluster_sep {bool} -- if True, returns also the cluster separation
                                        from the rest of the points (default: {False})
            include_score {bool} -- if True, returns also the score for each keyword
                                    in the clusters (default: {False})
        
        Returns:
            Tuple[List[keyword],Optional[float]] -- Tuple containing the cluster of keywords and the
                                                    cluster separation if include_cluster_sep == True
        """
        sorted_scores = sorted_scores[-(cluster_max_size + 1) :]
        diff_list = [
            (i + 1, sorted_scores[i + 1][1] - score[1])
            for i, score in enumerate(sorted_scores[:-1])
        ]
        ind_max_leap, maximum_leap = max(diff_list, key=lambda tup: tup[1])
        if maximum_leap > cluster_min_sensitivity:
            best_candidates = [
                ((kw, _score) if include_score else kw)
                for kw, _score in sorted_scores[ind_max_leap:]
            ]
        else:
            best_candidates = sorted_scores[-cluster_max_size:]
        if include_cluster_sep:
            return (best_candidates, maximum_leap)
        else:
            return (best_candidates,)

    def _sub_pred(
        self,
        ind,
        td_list,
        k=0,
        cluster_max_size=0,
        cluster_min_sensitivity=0,
        include_score=False,
        include_cluster_sep=False,
    ):
        """
        Sub-function used to parallelize the prediction.
        Specify either k for k-highest score mode or cluster_max_size for cluster mode

        Returns:
            List[Tuple[trapdoor,List[keyword]]] -- a list of tuples (trapdoor, predictions)
        """
        if bool(k) == bool(cluster_max_size) or (k and include_cluster_sep):
            raise ValueError("You have to choose either cluster mode or k-best mode")

        prediction = []
        for trapdoor in tqdm.tqdm(
            iterable=td_list,
            desc=f"Core {ind}: Evaluating each plain-cipher pairs",
            position=ind,
        ):
            try:
                trapdoor_ind = self.td_voc_info[trapdoor]["vector_ind"]
            except KeyError:
                logger.warning(f"Unknown trapdoor: {trapdoor}")
                prediction.append(
                    ((trapdoor, [], 0) if include_cluster_sep else (trapdoor, []))
                )
                continue
            trapdoor_vec = self.td_reduced_coocc[trapdoor_ind]

            score_list = []
            for keyword, kw_info in self.kw_voc_info.items(): 
                # Computes the matching with each keyword of the vocabulary extracted from similar documents
                keyword_vec = self.kw_reduced_coocc[kw_info["vector_ind"]]
                vec_diff = keyword_vec - trapdoor_vec
                # Distance between the keyword point and the trapdoor point in the known-queries sub-vector space
                td_kw_distance = self._norm(vec_diff)
                if td_kw_distance:
                    score = -np.log(td_kw_distance)
                else:  # If distance==0 => Perfect match
                    score = np.inf
                score_list.append((keyword, score))
            score_list.sort(key=lambda tup: tup[1])

            if cluster_max_size:  # Cluster mode
                cluster = self.__scores_to_cluster(
                    score_list,
                    cluster_max_size=cluster_max_size,
                    cluster_min_sensitivity=cluster_min_sensitivity,
                    include_score=include_score,
                    include_cluster_sep=include_cluster_sep,
                )
                prediction.append((trapdoor, *cluster))
            else:  # k-highest-score mode
                best_candidates = [
                    ((kw, _score) if include_score else kw)
                    for kw, _score in score_list[-k:]
                ]
                prediction.append((trapdoor, best_candidates))
        return prediction

    def predict(self, trapdoor_list, k=None):
        """
        Returns a prediction for each trapdoor in the list. No refinement. No clustering.
        Arguments:
            trapdoor_list {List[str]} -- List of the trapdoors to match with a keyword
        
        Keyword Arguments:
            k {int} -- the number of highest score keyword wanted for each trapdoor (default: {None})
        
        Returns:
            dict[trapdoor, predictions] -- dictionary with a list of k predictions for each trapdoor
        """
        if k is None:
            k = len(self.kw_voc_info)
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

    def predict_with_refinement(self, trapdoor_list, cluster_max_size=10, ref_speed=0):
        """Returns a cluster of predictions for each trapdoor using refinement.
        
        Arguments:
            trapdoor_list {List[str]} -- List of the trapdoors to match with a keyword
        
        Keyword Arguments:
            cluster_max_size {int} -- Maximum size of the clusters (default: {10})
            ref_speed {int} -- Refinement speed, i.e. number of queries "learnt" after each iteration (default: {0})
        
        Returns:
            dict[trapdoor, predictions] -- dictionary with a list of k predictions for each trapdoor
        """
        if ref_speed < 1:
            # Default refinement speed: 5% of the total number of trapdoors
            ref_speed = int(0.05 * len(self.td_voc_info))
        old_known = self._known_queries.copy()
        NUM_CORES = multiprocessing.cpu_count()
        local_td_list = list(trapdoor_list)

        final_results = []
        with poolcontext(processes=NUM_CORES) as pool, tqdm.tqdm(
            total=len(trapdoor_list), position=NUM_CORES, desc="Refining predictions"
        ) as pbar:
            while True:
                prev_td_nb = len(local_td_list)
                local_td_list = [
                    td for td in local_td_list if td not in self._known_queries.keys()
                ]  # Removes the known trapdoors
                pbar.update(prev_td_nb - len(local_td_list))
                pred_func = partial(
                    self._sub_pred,
                    cluster_max_size=cluster_max_size,
                    include_cluster_sep=True,
                )
                results = pool.starmap(  # Launch parallel predictions
                    pred_func,
                    enumerate([local_td_list[i::NUM_CORES] for i in range(NUM_CORES)]),
                )
                results = reduce(lambda x, y: x + y, results)

                # Extract the best preditions (must be single-point clusters)
                single_point_results = [tup for tup in results if len(tup[1]) == 1]
                single_point_results.sort(key=lambda tup: tup[2])

                if len(single_point_results) < ref_speed:  # Not enough single-point predictions 
                    final_results = [(td, cluster) for td, cluster, _sep in results]
                    break

                # Add the pseudo-known queries.
                new_known = {
                    td: list_candidates[-1]
                    for td, list_candidates, _sep in single_point_results[-ref_speed:]
                }
                self._known_queries.update(new_known)
                self.__refresh_reduced_coocc()

        # Concatenate known queries and last results
        prediction = {
            td: [kw] for td, kw in self._known_queries.items() if td in trapdoor_list
        }
        prediction.update(dict(final_results))

        # Reset the known queries
        self._known_queries = old_known
        self.__refresh_reduced_coocc()
        return prediction

    def accuracy(self, k=1, eval_dico=None):
        """Compute the base accuracy on the evaluation dictionary.
        No refinement. No clustering.
        
        Keyword Arguments:
            k {int} -- size of the prediction lists (default: {1})
            eval_dico {dict[trapdoor, keyword]} -- Evaluation dictionary (default: {None})
        
        Returns:
            Tuple[float, dict[trapdoor, List[keyword]]] -- a tuple containing the accuracy and the prediction dictionary
        """
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
        return recovery_rate, res_dict
