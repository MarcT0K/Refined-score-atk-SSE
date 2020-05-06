import argparse
import csv
import colorlog
import contextlib
import hashlib
import numpy as np
import tqdm

from src.common import KeywordExtractor, generate_known_queries, setup_logger
from src.email_extraction import split_df, extract_sent_mail_contents, extract_apache_ml
from src.query_generator import (
    QueryResultExtractor,
    ObfuscatedResultExtractor,
    PaddedResultExtractor,
)
from src.matchmaker import KeywordTrapdoorMatchmaker
from attack_scenario import DocumentSetExtraction

logger = colorlog.getLogger("Keyword Regression Attack")


def enron_result_generator(result_file="enron.csv"):
    """
    This function generates results of the ENRON attack testing several combination of parameters.

    This aims is to have a wide range of results and not to try every single possibility.
    """
    logger.handlers = []
    voc_size_possibilities = [500, 1000, 2000, 3000]
    range_int_array = lambda l, k, int_array: int_array[int_array < l * k]
    comb_voc_sizes = [
        (i, int(k * i), l)
        for i in voc_size_possibilities  # Voc size
        for k in [0.05, 0.1, 0.15, 0.25, 0.5]  # Seen trapdoor %
        for l in range_int_array(
            k, i, np.array([5, 10, 20, 40, 75, 100])
        )  # Nb of known trapdoors
    ]
    old_voc_size = old_queryset_size = 0
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Base acc",
            "Acc with cluster",
            "Acc with refinement",
            "Acc with refinement+cluster",
            "Cluster size",
            "Refinement speed",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        similar_docs, stored_docs = split_df(df=extract_sent_mail_contents(), frac=0.3)
        for (voc_size, queryset_size, nb_known_queries) in tqdm.tqdm(
            iterable=comb_voc_sizes,
            desc="Running tests with different combinations of parameters",
        ):
            cascade_change = False
            if voc_size != old_voc_size:
                similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
                real_extractor = QueryResultExtractor(stored_docs, voc_size, 1)
                cascade_change = True

            if queryset_size != old_queryset_size or cascade_change:
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=dict(query_voc).keys(),
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword, occ in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append((fake_trapdoor, occ))
                if known_queries.get(keyword):
                    temp_known[fake_trapdoor] = keyword
                eval_dico[fake_trapdoor] = keyword
            known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

            mm = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            td_list = list(set(eval_dico.keys()).difference(mm._known_queries.keys()))

            results = mm.predict(td_list, k=1)
            base_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            results = dict(mm._sub_pred(0, td_list, cluster_max_size=10))
            clust_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            ref_speed = int(0.05 * queryset_size)
            results = mm.predict_with_refinement(
                td_list, cluster_max_size=1, ref_speed=ref_speed
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            results = mm.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=ref_speed
            )
            clust_ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Base acc": "%.3f" % base_acc,
                    "Acc with cluster": "%.3f" % clust_acc,
                    "Acc with refinement": "%.3f" % ref_acc,
                    "Acc with refinement+cluster": "%.3f" % clust_ref_acc,
                    "Cluster size": 10,
                    "Refinement speed": ref_speed,
                }
            )
            old_voc_size, old_queryset_size = (voc_size, queryset_size)


def understand_variance(*args, **kwargs):
    """Procedure to simulate an inference attack.
    """
    setup_logger()
    # Params
    similar_voc_size = kwargs.get("similar_voc_size", 1000)
    server_voc_size = kwargs.get("server_voc_size", 1000)
    queryset_size = kwargs.get("queryset_size", 500)
    nb_known_queries = kwargs.get("nb_known_queries", int(queryset_size * 0.15))
    attack_dataset = kwargs.get("attack_dataset", "enron")
    include_most_frequent = bool(kwargs.get("include_most_frequent"))
    countermeasure = kwargs.get("countermeasure")
    logger.debug(f"Server vocabulary size: {server_voc_size}")
    logger.debug(f"Similar vocabulary size: {similar_voc_size}")
    if kwargs.get("L2"):
        logger.debug("L2 Scheme")
    else:
        logger.debug(f"L1 Scheme => Queryset size: {queryset_size}")

    try:
        extraction_procedure = DocumentSetExtraction[attack_dataset]
    except KeyError:
        raise ValueError("Unknown dataset")

    logger.info("ATTACK BEGINS")
    similar_docs, stored_docs = split_df(df=extraction_procedure(), frac=0.4)

    logger.info("Extracting keywords from similar documents.")
    similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)

    logger.info("Extracting keywords from stored documents.")

    if not countermeasure:
        real_extractor = QueryResultExtractor(stored_docs, server_voc_size, 1)
    elif countermeasure == "obfuscation":
        logger.debug("Obfuscation enabled")
        real_extractor = ObfuscatedResultExtractor(stored_docs, server_voc_size, 1)
    elif countermeasure == "padding":
        logger.debug("Padding enabled")
        real_extractor = PaddedResultExtractor(stored_docs, server_voc_size, 1)
    else:
        raise ValueError("Unknown countermeasure")

    accuracies = []
    for _i in range(kwargs.get("n_trials", 5)):
        logger.info(f"Generating {queryset_size} queries from stored documents")
        query_array, query_voc_plain = real_extractor.get_fake_queries(
            queryset_size, include_most_frequent=include_most_frequent
        )

        logger.debug(
            f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
        )

        known_queries = generate_known_queries(  # Extracted with uniform law
            similar_wordlist=similar_extractor.get_sorted_voc(),
            stored_wordlist=query_voc_plain,
            nb_queries=nb_known_queries - int(include_most_frequent),
        )
        if include_most_frequent:
            known_queries[query_voc_plain[0]] = query_voc_plain[0]

        logger.debug(
            "Hashing the keywords of the stored documents (transforming them into trapdoor tokens)"
        )
        # Trapdoor token == convey no information about the corresponding keyword
        temp_voc = []
        temp_known = {}
        eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
        for keyword in query_voc_plain:
            # We replace each keyword of the trapdoor dictionary by its hash
            # So the matchmaker truly ignores the keywords behind the trapdoors.
            fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
            temp_voc.append(fake_trapdoor)
            if known_queries.get(keyword):
                temp_known[fake_trapdoor] = keyword
            eval_dico[fake_trapdoor] = keyword
        query_voc = temp_voc
        known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

        matchmaker = KeywordTrapdoorMatchmaker(
            keyword_occ_array=similar_extractor.occ_array,
            keyword_sorted_voc=similar_extractor.get_sorted_voc(),
            trapdoor_occ_array=query_array,
            trapdoor_sorted_voc=query_voc,
            known_queries=known_queries,
        )
        base_acc = matchmaker.accuracy(k=1, eval_dico=eval_dico)[0]
        res_ref = matchmaker.predict_with_refinement(
            list(eval_dico.keys()), cluster_max_size=10, ref_speed=10
        )
        ref_acc = np.mean(
            [eval_dico[td] in candidates for td, candidates in res_ref.items()]
        )
        logger.info(f"Base accuracy: {base_acc} / Refinement accuracy: {ref_acc}")
        accuracies.append(ref_acc)

    return matchmaker, eval_dico, accuracies
