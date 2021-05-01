import multiprocessing

from functools import partial, reduce
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import colorlog
import numpy as np
import tqdm

from .common import poolcontext

logger = colorlog.getLogger("Refined Score attack")


class ScoreAttacker:
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc: List[str],
        known_queries: Dict[str, str],
        trapdoor_sorted_voc: Optional[List[str]],
        norm_ord=2,  # L2 (Euclidean norm),
        **kwargs,
    ):
        """Initialization of the matchmaker

        Arguments:
            keyword_occ_array {np.array} -- Keyword occurrence (row: similar documents; columns: keywords)
            trapdoor_occ_array {np.array} -- Trapdoor occurrence (row: stored documents; columns: trapdoors)
                                            the documents are unknown (just the identifier has
                                            been seen by the attacker)
            keyword_sorted_voc {List[str]} -- Keywoord vocabulary extracted from similar documents.
            known_queries {Dict[str, str]} -- Queries known by the attacker
            trapdoor_sorted_voc {Optional[List[str]]} -- The trapdoor voc can be a sorted list of hashes
                                                            to hide the underlying keywords.

        Keyword Arguments:
            norm_ord {int} -- Order of the norm used by the matchmaker (default: {2})
        """
        self.set_norm_ord(norm_ord=norm_ord)

        if not known_queries:
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Several trapdoors are linked to the same keyword.")

        self._known_queries = known_queries.copy()  # Keys: trapdoor, Values: keyword

        self.number_similar_docs = keyword_occ_array.shape[0]

        # NB: kw=KeyWord; td=TrapDoor
        self.kw_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(keyword_occ_array[:, ind])}
            for ind, word in enumerate(keyword_sorted_voc)
        }

        self.td_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(trapdoor_occ_array[:, ind])}
            for ind, word in enumerate(trapdoor_sorted_voc)
        }
        self.number_real_docs = self._estimate_nb_real_docs()
        for kw in self.kw_voc_info.keys():
            self.kw_voc_info[kw]["word_freq"] = (
                self.kw_voc_info[kw]["word_occ"] / self.number_similar_docs
            )
        for td in self.td_voc_info.keys():
            self.td_voc_info[td]["word_freq"] = (
                self.td_voc_info[td]["word_occ"] / self.number_real_docs
            )

        self._compute_coocc_matrices(keyword_occ_array, trapdoor_occ_array, **kwargs)

        self._refresh_reduced_coocc()

    def _compute_coocc_matrices(
        self, keyword_occ_array: np.array, trapdoor_occ_array: np.array
    ):
        logger.info("Computing cooccurrence matrices")
        # Can be improved using scipy's sparse matrices since coocc is symmetric
        self.kw_coocc = (
            np.dot(keyword_occ_array.T, keyword_occ_array) / self.number_similar_docs
        )
        np.fill_diagonal(self.kw_coocc, 0)

        self.td_coocc = (
            np.dot(trapdoor_occ_array.T, trapdoor_occ_array) / self.number_real_docs
        )
        np.fill_diagonal(self.td_coocc, 0)

    def _refresh_reduced_coocc(self):
        """Refresh the co-occurrence matrix based on the known queries."""
        ind_known_kw = [
            self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()
        ]
        self.kw_reduced_coocc = self.kw_coocc[:, ind_known_kw]
        ind_known_td = [
            self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()
        ]
        self.td_reduced_coocc = self.td_coocc[:, ind_known_td]

    def _estimate_nb_real_docs(self):
        """Estimates the number of documents stored."""
        nb_doc_ratio_estimator = np.mean(
            [
                self.td_voc_info[td]["word_occ"] / self.kw_voc_info[kw]["word_occ"]
                for td, kw in self._known_queries.items()
            ]
        )

        return self.number_similar_docs * nb_doc_ratio_estimator

    def set_norm_ord(self, norm_ord: int):
        """Set the order of the norm used to compute the scores.

        Arguments:
            norm_ord {int} -- norm order
        """
        self._norm = partial(np.linalg.norm, ord=norm_ord)

    @staticmethod
    def best_candidate_clustering(
        sorted_scores: List[Tuple[str, float]],
        cluster_max_size=1,
        cluster_min_sensitivity=0.0,
    ) -> Tuple[List[str], float]:
        """From a list of scores, extracts the best-candidate cluster Smax using
        simple-linkage clustering.

        Arguments:
            sorted_scores {List[Tuple[str, float]]} -- Sorted list of (keyword, score)

        Keyword Arguments:
            cluster_max_size {int} -- maximum size of the prediction clusters (default: {10})
            cluster_min_sensitivity {float} -- minimum leap size. Otherwise returns the
                                                maximum-size cluster (default: {0.0})

        Returns:
            Tuple[List[str],float] -- Tuple containing the cluster of keywords and the cluster separation
        """
        sorted_scores = sorted_scores[-(cluster_max_size + 1) :]
        diff_list = [
            (i + 1, sorted_scores[i + 1][1] - score[1])
            for i, score in enumerate(sorted_scores[:-1])
        ]
        ind_max_leap, maximum_leap = max(diff_list, key=lambda tup: tup[1])
        if np.isnan(maximum_leap):
            maximum_leap = 0
        if maximum_leap >= cluster_min_sensitivity:
            best_candidates = [kw for kw, _score in sorted_scores[ind_max_leap:]]
        else:
            best_candidates = sorted_scores[-cluster_max_size:]

        return (best_candidates, maximum_leap)

    def _sub_pred(
        self, _ind: int, td_list: List[str], cluster_max_size=1
    ) -> List[Tuple[str, List[str], float]]:
        """
        Sub-function used to parallelize the prediction.

        Returns:
            List[Tuple[str,List[str], float]] -- a list of tuples (trapdoor, [prediction], certainty) or
                                                    (trapdoor, cluster of predictions, certainty)
        """
        if cluster_max_size < 1:
            raise ValueError("The cluster size must be one or more.")

        prediction = []
        for trapdoor in td_list:
            try:
                trapdoor_ind = self.td_voc_info[trapdoor]["vector_ind"]
            except KeyError:
                logger.warning(f"Unknown trapdoor: {trapdoor}")
                prediction.append(
                    ((trapdoor, [], 0) if cluster_max_size > 1 else (trapdoor, [""], 0))
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

            if cluster_max_size > 1:  # Cluster mode
                kw_cluster, certainty = self.best_candidate_clustering(
                    score_list,
                    cluster_max_size=cluster_max_size,
                )
                prediction.append((trapdoor, kw_cluster, certainty))
            else:  # No clustering
                best_candidate = score_list[-1][0]
                certainty = score_list[-1][1] - score_list[-2][1]
                prediction.append((trapdoor, [best_candidate], certainty))
        return prediction

    def predict(self, trapdoor_list: List[str]) -> Dict[str, List[str]]:
        """
        Returns a prediction for each trapdoor in the list. No refinement. No clustering.
        Arguments:
            trapdoor_list {List[str]} -- List of the trapdoors to match with a keyword

        Returns:
            Dict[str, List[str]]-- dictionary with a one-prediction list for each trapdoor
        """
        predictions = {}
        nb_cores = multiprocessing.cpu_count()
        logger.info("Evaluating every possible keyword-trapdoor pair")
        with poolcontext(processes=nb_cores) as pool:
            pred_func = partial(self._sub_pred)
            results = pool.starmap(
                pred_func,
                enumerate([trapdoor_list[i::nb_cores] for i in range(nb_cores)]),
            )
            pred_list = reduce(lambda x, y: x + y, results)
            predictions = {td: kw for td, kw, _certainty in pred_list}
        return predictions

    def predict_with_refinement(
        self, trapdoor_list: List[str], cluster_max_size=1, ref_speed=0
    ) -> Dict[str, List[str]]:
        """Returns a cluster of predictions for each trapdoor using refinement.

        When cluster_max_size = 1, it corresponds to the Refined Score attack without clustering.

        Arguments:
            trapdoor_list {List[str]} -- List of the trapdoors to match with a keyword

        Keyword Arguments:
            cluster_max_size {int} -- Maximum size of the clusters (default: {1})
            ref_speed {int} -- Refinement speed, i.e. number of queries imputed at each iteration (default: {0})

        Returns:
            Dict[str, List[str]] -- dictionary with a prediction or cluster of predictions for each trapdoor
        """
        if ref_speed < 1:
            # Default refinement speed: 5% of the total number of trapdoors
            ref_speed = int(0.05 * len(self.td_voc_info))
        old_known = self._known_queries.copy()
        nb_cores = multiprocessing.cpu_count()
        unknown_td_list = list(trapdoor_list)

        final_results = []
        with poolcontext(processes=nb_cores) as pool, tqdm.tqdm(
            total=len(trapdoor_list), desc="Refining predictions"
        ) as pbar:
            while True:
                prev_td_nb = len(unknown_td_list)
                unknown_td_list = [
                    td for td in unknown_td_list if td not in self._known_queries.keys()
                ]  # Removes the known trapdoors
                pbar.update(prev_td_nb - len(unknown_td_list))
                pred_func = partial(
                    self._sub_pred,
                    cluster_max_size=cluster_max_size,
                )
                results = pool.starmap(  # Launch parallel predictions
                    pred_func,
                    enumerate([unknown_td_list[i::nb_cores] for i in range(nb_cores)]),
                )
                results = reduce(lambda x, y: x + y, results)

                # Extract the best preditions (must be single-point clusters)
                single_point_results = [tup for tup in results if len(tup[1]) == 1]
                single_point_results.sort(key=lambda tup: tup[2])

                if (
                    len(single_point_results) < ref_speed
                ):  # Not enough single-point predictions
                    final_results = [
                        (td, candidates) for td, candidates, _sep in results
                    ]
                    break

                # Add the pseudo-known queries.
                new_known = {
                    td: candidates[-1]
                    for td, candidates, _sep in single_point_results[-ref_speed:]
                }
                self._known_queries.update(new_known)
                self._refresh_reduced_coocc()

        # Concatenate known queries and last results
        prediction = {
            td: [kw] for td, kw in self._known_queries.items() if td in trapdoor_list
        }
        prediction.update(dict(final_results))

        # Reset the known queries
        self._known_queries = old_known
        self._refresh_reduced_coocc()
        return prediction


class GeneralizedScoreAttacker(ScoreAttacker):
    """
    This class is just a basic POC. There are significant improvements and refactoring possible.
    """

    def _compute_coocc_matrices(
        self, keyword_occ_array, trapdoor_occ_array, coocc_ord=2
    ):
        logger.info("Computing cooccurrence tensors")
        # Could be also improved if np.dot was implemented for N-dim (using tensors)
        self.kw_coocc = np.zeros([len(self.kw_voc_info)] * coocc_ord)
        self.__recursive_coocc_computer(keyword_occ_array, "keyword")
        self.kw_coocc = self.kw_coocc / self.number_similar_docs

        self.td_coocc = np.zeros([len(self.td_voc_info)] * coocc_ord)
        self.__recursive_coocc_computer(trapdoor_occ_array, "trapdoor")
        self.td_coocc = self.td_coocc / self.number_real_docs

    def _refresh_reduced_coocc(self):
        ind_known_kw = [
            self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()
        ]
        ind_known_td = [
            self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()
        ]
        slice_kw = np.ix_(
            np.arange(len(self.kw_coocc)), *[ind_known_kw] * (self.kw_coocc.ndim - 1)
        )
        slice_td = np.ix_(
            np.arange(len(self.td_coocc)), *[ind_known_td] * (self.td_coocc.ndim - 1)
        )
        self.kw_reduced_coocc = self.kw_coocc[slice_kw]
        self.td_reduced_coocc = self.td_coocc[slice_td]

    def __recursive_coocc_computer(
        self, index_mat, coocc_choice, possible_rows=None, previous_indices=None
    ):
        if coocc_choice not in ("keyword", "trapdoor"):
            raise ValueError

        coocc_mat = self.kw_coocc if coocc_choice == "keyword" else self.td_coocc
        if possible_rows is None:
            possible_rows = np.arange(index_mat.shape[0])
        if previous_indices is None:
            previous_indices = []
        max_loop = previous_indices[-1] + 1 if previous_indices else len(coocc_mat)

        if len(previous_indices) + 1 == coocc_mat.ndim:
            for i in range(max_loop):
                possible_permutations = permutations(previous_indices + [i])
                coocc = sum(index_mat[possible_rows, i])
                for perm in possible_permutations:
                    coocc_mat.itemset(perm, coocc)
        else:
            for i in range(max_loop):
                possible_row_offsets = np.argwhere(
                    index_mat[possible_rows, i]
                ).flatten()
                new_possible_rows = possible_rows[possible_row_offsets]
                self.__recursive_coocc_computer(
                    index_mat,
                    coocc_choice,
                    possible_rows=new_possible_rows,
                    previous_indices=previous_indices + [i],
                )
