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

logger = colorlog.getLogger("Keyword Regression Attack")


DocumentSetExtraction = {
    "enron": extract_sent_mail_contents,
    "apache": extract_apache_ml,
}


def attack_procedure(*args, **kwargs):
    """Procedure to simulate an inference attack.
    """
    setup_logger()
    # Params
    similar_voc_size = kwargs.get("similar_voc_size", 1000)
    server_voc_size = kwargs.get("server_voc_size", 1000)
    queryset_size = kwargs.get("queryset_size", 500)
    nb_known_queries = kwargs.get("nb_known_queries", int(queryset_size * 0.15))
    attack_dataset = kwargs.get("attack_dataset", "enron")
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

    ### KEYWORD EXTRACTION
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

    ### QUERY GENERATION
    logger.info(f"Generating {queryset_size} queries from stored documents")
    query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

    del real_extractor  # Reduce memory usage especially when applying countermeasures

    logger.debug(
        f"Picking {nb_known_queries} known queries ({nb_known_queries/queryset_size*100}% of the queries)"
    )
    known_queries = generate_known_queries(  # Extracted with uniform law
        similar_wordlist=similar_extractor.get_sorted_voc(),
        stored_wordlist=query_voc,
        nb_queries=nb_known_queries,
    )

    logger.debug(
        "Hashing the keywords of the stored documents (transforming them into trapdoor tokens)"
    )
    # Trapdoor token => convey no information about the corresponding keyword
    temp_voc = []
    temp_known = {}
    eval_dico = {}  # Keys: Trapdoor tokens; Values: Keywords
    for keyword in query_voc:
        # We replace each keyword of the trapdoor dictionary by its hash
        # So the matchmaker truly ignores the keywords behind the trapdoors.
        fake_trapdoor = hashlib.sha1(keyword.encode("utf-8")).hexdigest()
        temp_voc.append(fake_trapdoor)
        if known_queries.get(keyword):
            temp_known[fake_trapdoor] = keyword
        eval_dico[fake_trapdoor] = keyword
    query_voc = temp_voc
    known_queries = temp_known  # Keys: Trapdoor tokens; Values: Keywords

    ### THE INFERENCE ATTACK
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
    # NB: To be sure there is no bias in the algorithm we can compute the accuracy manually
    # as it is done for the refinement accuracy here.
    return matchmaker, eval_dico
    # The matchmaker and the eval_dico are returned so you
    # can run your own test in a Python terminal


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
        "--queryset-size",
        type=int,
        default=500,
        help="Number of queries which have been observed.",
    )
    parser.add_argument(
        "--nb-known-queries",
        type=int,
        default=50,
        help="Number of queries known by the attacker. Known Query=(Trapdoor, Corresponding Keyword)",
    )
    parser.add_argument(
        "--L2",
        default=False,
        action="store_true",
        help="Whether the server has an L2 scheme or not",
    )
    parser.add_argument(
        "--countermeasure",
        type=str,
        default="",
        help="Which countermeasure will be applied.",
    )
    parser.add_argument(
        "--attack-dataset",
        type=str,
        default="enron",
        help="Dataset used for the attack",
    )

    params = parser.parse_args()
    assert (
        params.nb_known_queries > 0 and params.nb_known_queries <= params.queryset_size
    )

    attack_procedure(**vars(params))
