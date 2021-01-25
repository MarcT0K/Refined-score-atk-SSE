"""Functions used to produce the results presented in the paper.
"""
import csv
import hashlib
import colorlog
import numpy as np

from queryvolution.src.common import KeywordExtractor, generate_known_queries
from queryvolution.src.email_extraction import (
    split_df,
    extract_sent_mail_contents,
    extract_apache_ml,
    extract_2_enron_mailboxes,
)
from queryvolution.src.query_generator import (
    QueryResultExtractor,
    ObfuscatedResultExtractor,
    PaddedResultExtractor,
)
from queryvolution.src.matchmaker import KeywordTrapdoorMatchmaker, GeneralMatchmaker


logger = colorlog.getLogger("QueRyvolution")
NB_REP = 50


def understand_variance(result_file="variance_understanding.csv"):
    similar_voc_size = 1000
    server_voc_size = 1000
    queryset_size = 150
    nb_known_queries = 5
    logger.debug(f"Server vocabulary size: {server_voc_size}")
    logger.debug(f"Similar vocabulary size: {similar_voc_size}")

    documents = extract_sent_mail_contents()

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Setup",
            "Nb similar docs",
            "Nb server docs",
            "Similar voc size",
            "Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "QueRyvolution Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for i in range(NB_REP):
            logger.info(f"Experiment {i+1} out of {NB_REP}")

            similar_docs, stored_docs = split_df(dframe=documents, frac=0.4)
            logger.info("Extracting keywords from similar documents.")
            similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)
            logger.info("Extracting keywords from stored documents.")
            real_extractor = QueryResultExtractor(stored_docs, server_voc_size, 1)

            logger.info(f"Generating {queryset_size} queries from stored documents")
            query_array, query_voc_plain = real_extractor.get_fake_queries(
                queryset_size
            )
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
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            res_ref = matchmaker.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=10
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in res_ref.items()]
            )
            writer.writerow(
                {
                    "Setup": "Normal",
                    "Nb similar docs": similar_docs.shape[0],
                    "Nb server docs": stored_docs.shape[0],
                    "Similar voc size": similar_voc_size,
                    "Server voc size": server_voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "QueRyvolution Acc": ref_acc,
                }
            )

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

            res_ref = matchmaker.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=10
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in res_ref.items()]
            )
            writer.writerow(
                {
                    "Setup": "Top 25%",
                    "Nb similar docs": similar_docs.shape[0],
                    "Nb server docs": stored_docs.shape[0],
                    "Similar voc size": similar_voc_size,
                    "Server voc size": server_voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "QueRyvolution Acc": ref_acc,
                }
            )


def cluster_size_statistics(result_file="cluster_size.csv"):
    # Params
    similar_voc_size = 1000
    server_voc_size = 1000
    queryset_size = 150
    nb_known_queries = 15
    max_cluster_sizes = [1, 5, 10, 20, 30, 50]

    cluster_results = {
        max_size: {"accuracies": [], "cluster_sizes": []}
        for max_size in max_cluster_sizes
    }

    for i in range(NB_REP):
        logger.info(f"Experiment {i+1} out of {NB_REP}")
        similar_docs, stored_docs = split_df(
            dframe=extract_sent_mail_contents(), frac=0.4
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

        for cluster_max_size in max_cluster_sizes:
            res_ref = matchmaker.predict_with_refinement(
                list(eval_dico.keys()), cluster_max_size=cluster_max_size, ref_speed=10
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in res_ref.items()]
            )
            clust_temp_sizes = [len(candidates) for candidates in res_ref.values()]
            cluster_results[cluster_max_size]["cluster_sizes"].extend(clust_temp_sizes)
            cluster_results[cluster_max_size]["accuracies"].append(ref_acc)

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Cluster maximum size",
            "Mean",
            "Median",
            "q0.6",
            "q0.7",
            "q0.75",
            "q0.8",
            "q0.85",
            "q0.9",
            "q0.95",
            "q0.99",
            "Average acc",
            "Cluster sizes",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for cluster_max_size, results in cluster_results.items():
            writer.writerow(
                {
                    "Cluster maximum size": cluster_max_size,
                    "Mean": np.mean(results["cluster_sizes"]),
                    "Median": np.quantile(results["cluster_sizes"], 0.5),
                    "q0.6": np.quantile(results["cluster_sizes"], 0.6),
                    "q0.7": np.quantile(results["cluster_sizes"], 0.7),
                    "q0.75": np.quantile(results["cluster_sizes"], 0.75),
                    "q0.8": np.quantile(results["cluster_sizes"], 0.8),
                    "q0.85": np.quantile(results["cluster_sizes"], 0.85),
                    "q0.9": np.quantile(results["cluster_sizes"], 0.9),
                    "q0.95": np.quantile(results["cluster_sizes"], 0.95),
                    "q0.99": np.quantile(results["cluster_sizes"], 0.99),
                    "Average acc": np.mean(results["accuracies"]),
                    "Cluster sizes": results["cluster_sizes"],
                }
            )


def base_results(result_file="base_attack.csv"):
    voc_size_possibilities = [500, 1000, 2000, 4000]
    experiment_params = [
        (i, j)
        for i in voc_size_possibilities  # Voc size
        for j in [15, 30, 60]  # known queries
        for _k in range(NB_REP)
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
        for (i, (voc_size, nb_known_queries)) in enumerate(experiment_params):
            logger.info(f"Experiment {i+1} out of {len(experiment_params)}")
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

            matchmaker = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            results = matchmaker.predict(td_list, k=1)
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


def attack_comparison(result_file="attack_comparison.csv"):
    experiment_params = [
        j for j in [5, 10, 20, 40] for _k in range(NB_REP)
    ]  # known queries

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
        for (i, nb_known_queries) in enumerate(experiment_params):
            logger.info(f"Experiment {i+1} out of {len(experiment_params)}")
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

            matchmaker = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            results = matchmaker.predict(td_list, k=1)
            base_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            results = dict(matchmaker._sub_pred(0, td_list, cluster_max_size=10))
            clust_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            ref_speed = int(0.05 * queryset_size)

            results = matchmaker.predict_with_refinement(
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


def document_set_results(result_file="document_set.csv"):
    email_extractors = [
        (extract_sent_mail_contents, "Enron"),
        (extract_apache_ml, "Apache"),
        (apache_reduced, "Apache reduced"),
    ]
    queryset_sizes = [i for i in [150, 300, 600, 1000] for _j in range(NB_REP)]

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
        i = 0
        for extractor, dataset in email_extractors:
            emails = extractor()
            for queryset_size in queryset_sizes:
                i += 1
                logger.info(
                    f"Experiment {i} out of {len(email_extractors)*len(queryset_sizes)}"
                )
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

                matchmaker = KeywordTrapdoorMatchmaker(
                    keyword_occ_array=similar_extractor.occ_array,
                    keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                    trapdoor_occ_array=query_array,
                    trapdoor_sorted_voc=td_voc,
                    known_queries=known_queries,
                )
                td_list = list(
                    set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
                )

                ref_speed = int(0.05 * queryset_size)
                results = matchmaker.predict_with_refinement(
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


def countermeasure_results(result_file="countermeasures.csv"):
    voc_size_possibilities = [500, 1000, 2000, 4000]
    experiment_params = [
        (i, j)
        for i in voc_size_possibilities  # Voc size
        for j in [
            QueryResultExtractor,
            ObfuscatedResultExtractor,
            PaddedResultExtractor,
        ]
        for _k in range(NB_REP)
    ]

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Countermeasure",
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "QueRyvolution acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        nb_known_queries = 15
        for (i, (voc_size, query_extractor)) in enumerate(experiment_params):
            logger.info(f"Experiment {i+1} out of {len(experiment_params)}")
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)
            queryset_size = int(voc_size * 0.15)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = query_extractor(stored_docs, voc_size, 1)

            query_array, query_voc = real_extractor.get_fake_queries(queryset_size)
            del real_extractor

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

            matchmaker = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            ref_speed = int(0.05 * queryset_size)
            results = matchmaker.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=ref_speed
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Countermeasure": str(query_extractor),
                    "Nb similar docs": similar_docs.shape[0],
                    "Nb server docs": stored_docs.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "QueRyvolution acc": ref_acc,
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
            "QueRyvolution acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        voc_size = 300
        queryset_size = 75
        nb_known_queries = 10

        for _i in range(NB_REP):
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

            matchmaker = GeneralMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
                coocc_ord=3,
            )
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            results = matchmaker.predict_with_refinement(
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
                    "QueRyvolution acc": ref_acc,
                }
            )

        for _i in range(NB_REP):  # TODO merge with previous loop
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

            matchmaker = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            results = matchmaker.predict_with_refinement(
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
                    "QueRyvolution acc": ref_acc,
                }
            )


def query_distrib_results(result_file="query_distrib.csv"):
    voc_size_possibilities = [1000, 2000, 4000]
    experiment_params = [
        (i, j)
        for i in voc_size_possibilities  # Voc size
        for j in ["uniform", "zipfian", "inv_zipfian"]
        for _k in range(NB_REP)
    ]

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Query distribution",
            "Nb similar docs",
            "Nb server docs",
            "Similar/Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "QueRyvolution acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        enron = extract_sent_mail_contents()
        nb_known_queries = 15
        for (i, (voc_size, query_distrib)) in enumerate(experiment_params):
            logger.info(f"Experiment {i+1} out of {len(experiment_params)}")
            similar_docs, stored_docs = split_df(dframe=enron, frac=0.4)
            queryset_size = int(voc_size * 0.15)

            similar_extractor = KeywordExtractor(similar_docs, voc_size, 1)
            real_extractor = QueryResultExtractor(
                stored_docs, voc_size, 1, distribution=query_distrib
            )

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

            matchmaker = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.get_sorted_voc(),
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            ref_speed = int(0.05 * queryset_size)
            results = matchmaker.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=ref_speed
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in results.items()]
            )

            writer.writerow(
                {
                    "Query distribution": query_distrib,
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar/Server voc size": voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "QueRyvolution acc": ref_acc,
                }
            )


def usecase_example(result_file="usecase.csv"):
    similar_voc_size = 500
    server_voc_size = 500
    queryset_size = 150
    nb_known_queries = 15

    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Similar voc size",
            "Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "QueRyvolution Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for i in range(NB_REP):
            logger.info(f"Experiment {i+1} out of {NB_REP}")

            similar_docs, stored_docs = extract_2_enron_mailboxes()
            logger.info("Extracting keywords from similar documents.")
            similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)
            logger.info("Extracting keywords from stored documents.")
            real_extractor = QueryResultExtractor(stored_docs, server_voc_size, 1)

            logger.info(f"Generating {queryset_size} queries from stored documents")
            query_array, query_voc_plain = real_extractor.get_fake_queries(
                queryset_size
            )
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
            td_list = list(
                set(eval_dico.keys()).difference(matchmaker._known_queries.keys())
            )

            res_ref = matchmaker.predict_with_refinement(
                td_list, cluster_max_size=10, ref_speed=10
            )
            ref_acc = np.mean(
                [eval_dico[td] in candidates for td, candidates in res_ref.items()]
            )
            logger.info(f"QueRyvolution accuracy: {ref_acc}")
            writer.writerow(
                {
                    "Nb similar docs": similar_docs.shape[0],
                    "Nb server docs": stored_docs.shape[0],
                    "Similar voc size": similar_voc_size,
                    "Server voc size": server_voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "QueRyvolution Acc": ref_acc,
                }
            )
