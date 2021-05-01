import argparse
import hashlib

import colorlog
import numpy as np

from src.common import (
    KeywordExtractor,
    generate_known_queries,
    setup_logger,
)
from src.email_extraction import (
    split_df,
    extract_sent_mail_contents,
    extract_apache_ml,
)
from src.query_generator import (
    QueryResultExtractor,
    ObfuscatedResultExtractor,
    PaddedResultExtractor,
)
from src.attackers import ScoreAttacker

logger = colorlog.getLogger("Refined Score attack")


DocumentSetExtraction = {
    "enron": extract_sent_mail_contents,
    "apache": extract_apache_ml,
}


def attack_procedure(*args, **kwargs):
    """Procedure to simulate a refined score attack."""
    setup_logger()
    # Params
    logger.debug("PARAMETERS:")
    similar_voc_size = kwargs.get("similar_voc_size", 1200)
    logger.debug(f"\t- Similar vocabulary size: {similar_voc_size}")
    server_voc_size = kwargs.get("server_voc_size", 1000)
    logger.debug(f"\t- Server vocabulary size: {server_voc_size}")
    queryset_size = kwargs.get("queryset_size", int(0.15 * server_voc_size))
    logger.debug(f"\t- Query set size: {queryset_size}")
    nb_known_queries = kwargs.get("nb_known_queries", 10)
    logger.debug(f"\t- Number of known queries: {nb_known_queries}")
    attack_dataset = kwargs.get("attack_dataset", "enron")
    countermeasure = kwargs.get("countermeasure")

    try:
        extraction_procedure = DocumentSetExtraction[attack_dataset]
    except KeyError:
        raise ValueError("Unknown dataset")
    logger.debug(f"\t- Dataset: {attack_dataset}")

    logger.info("ATTACK BEGINS")
    similar_docs, stored_docs = split_df(dframe=extraction_procedure(), frac=0.4)

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

    ### ATTACK
    attacker = ScoreAttacker(
        keyword_occ_array=similar_extractor.occ_array,
        keyword_sorted_voc=similar_extractor.get_sorted_voc(),
        trapdoor_occ_array=query_array,
        trapdoor_sorted_voc=query_voc,
        known_queries=known_queries,
    )

    td_list = list(set(eval_dico.keys()).difference(attacker._known_queries.keys()))

    results = attacker.predict(td_list)
    base_acc = np.mean(
        [eval_dico[td] == prediction[0] for td, prediction in results.items()]
    )

    results = attacker.predict_with_refinement(td_list, ref_speed=10)
    ref_acc = np.mean(
        [eval_dico[td] == prediction[0] for td, prediction in results.items()]
    )

    logger.info(f"Base accuracy: {base_acc} / Refinement accuracy: {ref_acc}")
    return attacker, eval_dico
    # The matchmaker and the eval_dico are returned so you
    # can run your own test in a Python terminal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score attack simulator")
    parser.add_argument(
        "--similar-voc-size",
        type=int,
        default=1200,
        help="Size of the vocabulary extracted from similar documents.",
    )
    parser.add_argument(
        "--server-voc-size",
        type=int,
        default=1000,
        help="Size of the 'queryable' vocabulary.",
    )
    parser.add_argument(
        "--queryset-size",
        type=int,
        default=150,
        help="Number of queries which have been observed.",
    )
    parser.add_argument(
        "--nb-known-queries",
        type=int,
        default=10,
        help="Number of queries known by the attacker. Known Query=(Trapdoor, Corresponding Keyword)",
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
        params.nb_known_queries > 0
        and params.nb_known_queries <= params.queryset_size
        and params.queryset_size < params.server_voc_size
        and params.similar_voc_size > 0
    )

    attack_procedure(**vars(params))
