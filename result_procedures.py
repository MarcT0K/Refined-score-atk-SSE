"""Functions used to produce the results presented in the paper.
"""
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
from src.matchmaker import KeywordTrapdoorMatchmaker, GeneralMatchmaker
from attack_scenario import DocumentSetExtraction

logger = colorlog.getLogger("QueRyvolution")


def understand_variance(*args, **kwargs):
    """Procedure to simulate an inference attack.
    """
    setup_logger()
    # Params
    similar_voc_size = kwargs.get("similar_voc_size", 1000)
    server_voc_size = kwargs.get("server_voc_size", 1000)
    queryset_size = kwargs.get("queryset_size", 150)
    nb_known_queries = kwargs.get("nb_known_queries", 5)
    attack_dataset = kwargs.get("attack_dataset", "enron")
    countermeasure = kwargs.get("countermeasure")
    logger.debug(f"Server vocabulary size: {server_voc_size}")
    logger.debug(f"Similar vocabulary size: {similar_voc_size}")
    if kwargs.get("L2"):
        logger.debug("L2 Scheme")
    else:
        logger.debug(
            f"L1 Scheme => Queryset size: {queryset_size}, Known queries: {nb_known_queries}"
        )

    try:
        extraction_procedure = DocumentSetExtraction[attack_dataset]
    except KeyError:
        raise ValueError("Unknown dataset")

    logger.info("ATTACK BEGINS")
    similar_docs, stored_docs = split_df(dframe=extraction_procedure(), frac=0.4)

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

    logger.info(f"Generating {queryset_size} queries from stored documents")
    query_array, query_voc_plain = real_extractor.get_fake_queries(queryset_size)
    del real_extractor
    accuracies = []
    for _i in range(kwargs.get("n_trials", 40)):
        logger.debug(
            f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
        )

        known_queries = generate_known_queries(  # Extracted with uniform law
            similar_wordlist=similar_extractor.get_sorted_voc(),
            stored_wordlist=query_voc_plain,
            nb_queries=nb_known_queries,
        )

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
        td_list = list(
            set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
        )

        results = matchmaker.predict(td_list, k=1)
        base_acc = np.mean(
            [eval_dico[td] in candidates for td, candidates in results.items()]
        )
        res_ref = matchmaker.predict_with_refinement(
            td_list, cluster_max_size=10, ref_speed=10
        )
        ref_acc = np.mean(
            [eval_dico[td] in candidates for td, candidates in res_ref.items()]
        )
        logger.info(f"Base accuracy: {base_acc} / Refinement accuracy: {ref_acc}")
        accuracies.append(ref_acc)

    accuracies_frequent = []
    for _i in range(kwargs.get("n_trials", 40)):
        logger.debug(
            f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
        )

        stored_wordlist = query_voc_plain[: len(query_voc_plain) // 4]

        known_queries = generate_known_queries(  # Extracted with uniform law
            similar_wordlist=similar_extractor.get_sorted_voc(),
            stored_wordlist=stored_wordlist,
            nb_queries=nb_known_queries,
        )

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
        td_list = list(
            set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
        )

        results = matchmaker.predict(td_list, k=1)
        base_acc = np.mean(
            [eval_dico[td] in candidates for td, candidates in results.items()]
        )
        res_ref = matchmaker.predict_with_refinement(
            td_list, cluster_max_size=10, ref_speed=10
        )
        ref_acc = np.mean(
            [eval_dico[td] in candidates for td, candidates in res_ref.items()]
        )
        logger.info(f"Base accuracy: {base_acc} / Refinement accuracy: {ref_acc}")
        accuracies_frequent.append(ref_acc)

    return accuracies, accuracies_frequent


def mean_cluster_size(*args, **kwargs):
    setup_logger()
    # Params
    similar_voc_size = 1000
    server_voc_size = 1000
    queryset_size = 150
    nb_known_queries = 15

    cluster_sizes = []
    for _i in range(kwargs.get("n_trials", 10)):
        similar_docs, stored_docs = split_df(
            dframe=extract_sent_mail_contents, frac=0.4
        )

        logger.info("Extracting keywords from similar documents.")
        similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)

        logger.info("Extracting keywords from stored documents.")

        real_extractor = QueryResultExtractor(stored_docs, server_voc_size, 1)

        logger.info(f"Generating {queryset_size} queries from stored documents")
        query_array, query_voc_plain = real_extractor.get_fake_queries(queryset_size)
        del real_extractor
        logger.debug(
            f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
        )

        known_queries = generate_known_queries(  # Extracted with uniform law
            similar_wordlist=similar_extractor.get_sorted_voc(),
            stored_wordlist=query_voc_plain,
            nb_queries=nb_known_queries,
        )

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

        res_ref = matchmaker.predict_with_refinement(
            list(eval_dico.keys()), cluster_max_size=10, ref_speed=10
        )
        ref_acc = np.mean(
            [eval_dico[td] in candidates for td, candidates in res_ref.items()]
        )
        clust_temp_sizes = [len(candidates) for candidates in res_ref.values()]
        cluster_sizes.extend(clust_temp_sizes)
    return cluster_sizes


def base_results(result_file="base.csv"):
    setup_logger()
    voc_size_possibilities = [500, 1000, 2000, 4000]
    comb_voc_sizes = [
        (i, j)
        for i in voc_size_possibilities  # Voc size
        for j in [15, 30, 60]  # known queries
        for _k in range(5)
    ]

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Base acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        for voc_size, nb_known_queries in tqdm.tqdm(
            iterable=comb_voc_sizes,
            desc="Running tests with different combinations of parameters",
        ):
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)
            queryset_size = int(voc_size * 0.15)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = QueryResultExtractor(stored_docs, voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append(fake_trapdoor)
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

            writer.writerow(
                {
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Base acc": base_acc,
                }
            )


def comp_results(result_file="comp.csv"):
    setup_logger()
    comb_voc_sizes = [j for j in [5, 10, 20, 40] for _k in range(10)]  # known queries

    similar_voc_size = 1200
    real_voc_size = 1000
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Similar voc size",
            "Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Base acc",
            "Acc with cluster",
            "Acc with refinement",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        for nb_known_queries in tqdm.tqdm(
            iterable=comb_voc_sizes,
            desc="Running tests with different combinations of parameters",
        ):
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)
            queryset_size = int(real_voc_size * 0.15)

            similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)
            real_extractor = QueryResultExtractor(stored_docs, real_voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append(fake_trapdoor)
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
                td_list, cluster_max_size=10, ref_speed=ref_speed
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar voc size": similar_voc_size,
                    "Server voc size": real_voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Base acc": base_acc,
                    "Acc with cluster": clust_acc,
                    "Acc with refinement": ref_acc,
                }
            )


def apache_reduced():
    ratio = 30109 / 50878
    apache_full = extract_apache_ml()
    apache_red, _ = split_df(apache_full, ratio)
    return apache_red


def ref_results(result_file="ref.csv"):
    setup_logger()
    email_extractors = [
        (extract_sent_mail_contents, "Enron"),
        (extract_apache_ml, "Apache"),
        (apache_reduced, "Apache reduced"),
    ]
    queryset_sizes = [i for i in [150, 300, 600, 1000] for _j in range(10)]

    similar_voc_size = 1000
    real_voc_size = 1000
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Dataset",
            "Nb similar docs",
            "Nb server docs",
            "Similar voc size",
            "Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for extractor, dataset in email_extractors:
            emails = extractor()
            for queryset_size in tqdm.tqdm(
                iterable=queryset_sizes,
                desc="Running tests with different combinations of parameters",
            ):
                similar_docs, stored_docs = split_df(dframe=emails, frac=0.4)

                similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)
                real_extractor = QueryResultExtractor(stored_docs, real_voc_size, 1)

                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

                known_queries = generate_known_queries(
                    similar_wordlist=similar_extractor.get_sorted_voc(),
                    stored_wordlist=query_voc,
                    nb_queries=15,
                )

                td_voc = []
                temp_known = {}
                eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
                for keyword in query_voc:
                    fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                    td_voc.append(fake_trapdoor)
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
                td_list = list(
                    set(eval_dico.keys()).difference(mm._known_queries.keys())
                )

                ref_speed = int(0.05 * queryset_size)
                results = mm.predict_with_refinement(
                    td_list, cluster_max_size=10, ref_speed=ref_speed
                )
                ref_acc = np.mean(
                    [eval_dico[td] in candidates for td, candidates in results.items()]
                )

                writer.writerow(
                    {
                        "Dataset": dataset,
                        "Nb similar docs": similar_extractor.occ_array.shape[0],
                        "Nb server docs": real_extractor.occ_array.shape[0],
                        "Similar voc size": similar_voc_size,
                        "Server voc size": real_voc_size,
                        "Nb queries seen": queryset_size,
                        "Nb queries known": 15,
                        "Acc": ref_acc,
                    }
                )


def ref_voc_size_results(result_file="ref_voc_size.csv"):
    setup_logger()
    voc_size_possibilities = [1000, 1500, 2000, 3000, 4000]
    comb_voc_sizes = [i for i in voc_size_possibilities for _k in range(5)]  # Voc size

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Ref acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        nb_known_queries = 20
        for voc_size in tqdm.tqdm(
            iterable=comb_voc_sizes,
            desc="Running tests with different combinations of parameters",
        ):
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)
            queryset_size = int(voc_size * 0.15)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = QueryResultExtractor(stored_docs, voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append(fake_trapdoor)
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

            ref_speed = int(0.05 * queryset_size)
            results = mm.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=ref_speed
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Ref acc": ref_acc,
                }
            )


def countermeasure_results(result_file="countermeasures.csv"):
    setup_logger()
    voc_size_possibilities = [500, 1000, 2000, 4000]
    comb_voc_sizes = [
        (i, j)
        for i in voc_size_possibilities  # Voc size
        for j in [
            QueryResultExtractor,
            ObfuscatedResultExtractor,
            PaddedResultExtractor,
        ]
        for _k in range(5)
    ]

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Countermeasure",
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Ref acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        nb_known_queries = 15
        for voc_size, query_extractor in tqdm.tqdm(
            iterable=comb_voc_sizes,
            desc="Running tests with different combinations of parameters",
        ):
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)
            queryset_size = int(voc_size * 0.15)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = query_extractor(stored_docs, voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append(fake_trapdoor)
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

            ref_speed = int(0.05 * queryset_size)
            results = mm.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=ref_speed
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Countermeasure": str(query_extractor),
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Ref acc": ref_acc,
                }
            )


def generalization(result_file="generalization.csv"):
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Generalized version",
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Ref acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        voc_size = 300
        queryset_size = 75
        nb_known_queries = 10

        for _i in range(10):
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = QueryResultExtractor(stored_docs, voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append(fake_trapdoor)
                if known_queries.get(keyword):
                    temp_known[fake_trapdoor] = keyword
                eval_dico[fake_trapdoor] = keyword
            known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

            mm = GeneralMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
                coocc_ord=3,
            )
            td_list = list(set(eval_dico.keys()).difference(mm._known_queries.keys()))

            results = mm.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=5
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )
            writer.writerow(
                {
                    "Generalized version": True,
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Ref acc": ref_acc,
                }
            )

        for _i in range(10):
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = QueryResultExtractor(stored_docs, voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                similar_wordlist=similar_extractor.get_sorted_voc(),
                stored_wordlist=query_voc,
                nb_queries=nb_known_queries,
            )

            td_voc = []
            temp_known = {}
            eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
            for keyword in query_voc:
                fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
                td_voc.append(fake_trapdoor)
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

            results = mm.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=5
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Generalized version": False,
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Ref acc": ref_acc,
                }
            )
