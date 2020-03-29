import argparse
import csv
import colorlog
import contextlib
import hashlib
import numpy as np
import tqdm

from src.common import KeywordExtractor, generate_known_queries, setup_logger
from src.email_extraction import split_df, extract_sent_mail_contents
from src.query_generator import QueryResultExtractor
from src.matchmaker import KeywordTrapdoorMatchmaker

logger = colorlog.getLogger("Keyword Regression Attack")


def attack_enron(*args, **kwargs):
    setup_logger()
    # Params
    similar_voc_size = kwargs.get("similar_voc_size", 1000)
    server_voc_size = kwargs.get("server_voc_size", 1000)
    queryset_size = kwargs.get("queryset_size", 500)
    nb_known_queries = kwargs.get("nb_known_queries", int(queryset_size*0.15))
    logger.debug(f"Server vocabulary size: {server_voc_size}")
    logger.debug(f"Similar vocabulary size: {similar_voc_size}")
    if kwargs.get("L1"):
        logger.debug("L1 Scheme")
    else:
        logger.debug(f"L0 Scheme => Queryset size: {queryset_size}")

    logger.info("ATTACK BEGINS")

    similar_docs, stored_docs = split_df(df=extract_sent_mail_contents(), frac=0.4)

    logger.info("Extracting keywords from similar documents.")
    similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)

    logger.info("Extracting keywords from stored documents.")
    real_extractor = QueryResultExtractor(stored_docs, server_voc_size, 1)
    logger.info(f"Generating {queryset_size} queries from stored documents")
    query_array, query_voc = real_extractor.get_fake_queries(
        queryset_size
    )  # Extracted with zipfian law

    logger.debug(
        f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
    )
    known_queries = generate_known_queries(  # Extracted with uniform law
        plain_wordlist=dict(similar_extractor.sorted_voc).keys(),
        trapdoor_wordlist=dict(query_voc).keys(),
        nb_queries=nb_known_queries,
    )

    logger.debug(
        "Hashing the keywords of the stored documents (transforming them into trapdoor tokens)"
    )
    # Trapdoor token == convey no information about the corresponding keyword
    temp_voc = []
    temp_known = {}
    eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
    for keyword, occ in query_voc:
        fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
        temp_voc.append((fake_trapdoor, occ))
        if known_queries.get(keyword):
            temp_known[fake_trapdoor] = keyword
        eval_dico[fake_trapdoor] = keyword
    query_voc = temp_voc
    known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

    matchmaker = KeywordTrapdoorMatchmaker(
        keyword_occ_array=similar_extractor.occ_array,
        keyword_sorted_voc=similar_extractor.sorted_voc,
        trapdoor_occ_array=query_array,
        trapdoor_sorted_voc=query_voc,
        known_queries=known_queries,
    )
    logger.info(f"Accuracy: {matchmaker.accuracy(k=1, eval_dico=eval_dico)[0]}")
    return matchmaker, eval_dico


def enron_result_generator(result_file="enron.csv"):
    """
    This function generates results of the ENRON attack testing several combination of parameters.

    This aims is to have a wide range of results and not to try every single possibility.
    """
    logger.handlers = []
    voc_size_possibilities = [500, 1000, 2000, 4000]
    range_int_array = lambda l, k, int_array: int_array[int_array < l * k]
    comb_voc_sizes = [
        (i, j, int(k * i), l)
        for ind, i in enumerate(voc_size_possibilities)  # Server voc size
        for j in voc_size_possibilities[ind:]  # Similar voc size
        for k in [0.05, 0.1, 0.15, 0.25, 0.5]  # Seen trapdoor %
        for l in range_int_array(
            k, i, np.array([10, 20, 40, 70, 100, 200])
        )  # Nb of known trapdoors
    ]
    old_similar_voc_size = old_server_voc_size = old_queryset_size = 0
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Similar voc size",
            "Server voc size",
            "Nb queries seen",
            "Nb queries known",
            "Accuracy",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        similar_docs, stored_docs = split_df(df=extract_sent_mail_contents(), frac=0.3)
        for (
            server_voc_size,
            similar_voc_size,
            queryset_size,
            nb_known_queries,
        ) in tqdm.tqdm(
            iterable=comb_voc_sizes,
            desc="Running tests with different combinations of parameters",
        ):
            cascade_change = False
            if similar_voc_size != old_similar_voc_size:
                similar_extractor = KeywordExtractor(similar_docs, similar_voc_size, 1)
                cascade_change = True

            if server_voc_size != old_server_voc_size:
                real_extractor = QueryResultExtractor(stored_docs, server_voc_size, 1)
                cascade_change = True

            if queryset_size != old_queryset_size or cascade_change:
                query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

            known_queries = generate_known_queries(
                plain_wordlist=dict(similar_extractor.sorted_voc).keys(),
                trapdoor_wordlist=dict(query_voc).keys(),
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

            matchmaker = KeywordTrapdoorMatchmaker(
                keyword_occ_array=similar_extractor.occ_array,
                keyword_sorted_voc=similar_extractor.sorted_voc,
                trapdoor_occ_array=query_array,
                trapdoor_sorted_voc=td_voc,
                known_queries=known_queries,
            )
            acc = matchmaker.accuracy(k=1, eval_dico=eval_dico)[0]

            writer.writerow(
                {
                    "Nb similar docs": similar_extractor.occ_array.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Similar voc size": similar_voc_size,
                    "Server voc size": server_voc_size,
                    "Nb queries seen": queryset_size,
                    "Nb queries known": nb_known_queries,
                    "Accuracy": acc,
                }
            )
            old_similar_voc_size, old_server_voc_size, old_queryset_size = (
                similar_voc_size,
                server_voc_size,
                queryset_size,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--similar-voc-size",
        type=int,
        default=1000,
        help="Size of the vocabulary extracted from similar documents.",
    )
    parser.add_argument(
        "--server-voc-size",
        type=int,
        default=1000,
        help="Size of the vocabulary stored in the server.",
    )
    parser.add_argument(
        "--queryset-size", type=int, default=500, help="Fake queryset size"
    )
    parser.add_argument(
        "--nb-known-queries",
        type=int,
        default=50,
        help="Number of queries known by the attacker. Known Query=(Trapdoor, Corresponding Keyword)",
    )
    parser.add_argument(
        "--L1",
        default=False,
        action="store_true",
        help="Whether the server has an L1 scheme or not",
    )
    parser.add_argument(
        "--attack-dataset",
        type=str,
        default="enron",
        help="Dataset used for the attack",
    )

    params = parser.parse_args()
    assert params.nb_known_queries > 0 and params.nb_known_queries <= params.queryset_size
    if (
        params.attack_dataset == "enron"
    ):  # We can use an enum or a dict if we have a lot of possible dataset
        attack_enron(**vars(params))
    elif params.attack_dataset == "apache":
        raise NotImplementedError
    else:
        raise ValueError("Unknown dataset")
