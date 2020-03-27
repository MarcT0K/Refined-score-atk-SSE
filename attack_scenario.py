import argparse
import colorlog

from src.common import (
    KeywordExtractor,
    generate_known_queries,
    setup_logger,
    sorted_to_dict,
)
from src.email_extraction import split_df, extract_sent_mail_contents
from src.query_generator import QueryResultExtractor
from src.assigner import PlainCipherAssigner

logger = colorlog.getLogger("Keyword Regression Attack")


def attack_enron(*args, **kwargs):
    setup_logger()
    # Params
    plaintext_voc_size = kwargs.get("plaintext_voc_size", 1000)
    ciphertext_voc_size = kwargs.get("ciphertext_voc_size", 1000)
    queryset_size = kwargs.get("queryset_size", 500)
    known_queries_ratio = kwargs.get("known_queries_ratio", 0.15)
    logger.debug(f"Ciphertext vocabulary size: {ciphertext_voc_size}")
    logger.debug(f"Plaintext vocabulary size: {plaintext_voc_size}")
    if kwargs.get("L1"):
        logger.debug("L1 Scheme")
    else:
        logger.debug(f"L0 Scheme => Queryset size: {queryset_size}")

    logger.info("ATTACK BEGINS")

    similar_docs, stored_docs = split_df(df=extract_sent_mail_contents(), frac=0.4)

    logger.info("Extracting keywords from similar documents.")
    similar_extractor = KeywordExtractor(similar_docs, plaintext_voc_size, 1)

    logger.info("Extracting keywords from stored documents.")
    real_extractor = QueryResultExtractor(stored_docs, ciphertext_voc_size, 1)
    query_array, query_voc = real_extractor.get_fake_queries(queryset_size)

    assign = PlainCipherAssigner(
        plain_occ_array=similar_extractor.occ_array,
        plain_sorted_voc=similar_extractor.sorted_voc,
        cipher_occ_array=query_array,
        cipher_sorted_voc=query_voc,
        known_queries=generate_known_queries(
            plain_wordlist=sorted_to_dict(similar_extractor.sorted_voc).keys(),
            trapdoor_wordlist=sorted_to_dict(query_voc).keys(),
            nb_queries=int(queryset_size * known_queries_ratio),
        ),
    )

    return assign


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--plaintext-voc-size", type=int, default=1000, help="Plaintext vocabulary size"
    )
    parser.add_argument(
        "--ciphertext-voc-size",
        type=int,
        default=1000,
        help="Ciphertext vocabulary size",
    )
    parser.add_argument(
        "--queryset-size", type=int, default=500, help="Fake queryset size"
    )
    parser.add_argument(
        "--known-queries-ratio",
        type=float,
        default=0.15,
        help="Ratio of queries known by the attacker",
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
    # TODO: parser.add_argument("--vector-size", type=int, default=50)
    # Or do a kind of validation to find a convenient vector size

    params = parser.parse_args()
    assert 0 < params.known_queries_ratio and params.known_queries_ratio <= 1
    if (
        params.attack_dataset == "enron"
    ):  # We can use an enum or a dict if we have a lot of possible dataset
        attack_enron(**vars(params))
    else:
        raise ValueError("Unknown dataset")
