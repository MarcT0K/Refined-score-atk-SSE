import argparse
import colorlog

from src.common import KeywordExtractor, GloveWordEmbedding, setup_logger
from src.email_extraction import split_df, extract_sent_mail_contents
from src.query_generator import QueryResultExtractor

logger = colorlog.getLogger("Keyword Alignment Attack")


def plaintext_embedding_phase(corpus_df, voc_size=100, minimum_freq=1):
    logger.info("START PLAINTEXT EMBEDDING PHASE")
    extractor = KeywordExtractor(corpus_df, voc_size, minimum_freq)

    word_embedder = GloveWordEmbedding(
        vocab_filename="plain_voc.txt",
        vector_filename="plain",
        occ_array=extractor.occ_array,
        voc_with_occ=extractor.sorted_voc,
    )
    word_embedder()
    word_embedder.generate_useless_muse_dico()
    logger.info("END PLAINTEXT EMBEDDING PHASE")
    return word_embedder.coocc_mat, word_embedder.sorted_voc


def ciphertext_embedding_phase(
    corpus_df, voc_size=100, minimum_freq=1, queryset_size=1000, L1=False
):
    logger.info("START CIPHERTEXT EMBEDDING PHASE")
    extractor = QueryResultExtractor(corpus_df, voc_size, minimum_freq)

    if L1:
        word_embedder = GloveWordEmbedding(
            vocab_filename="cipher_voc.txt",
            vector_filename="cipher",
            occ_array=extractor.occ_array,
            voc_with_occ=extractor.sorted_voc,
        )
    else:
        word_embedder = GloveWordEmbedding(
            vocab_filename="cipher_voc.txt",
            vector_filename="cipher",
            query_ans_dict=extractor.get_query_answer(size=queryset_size),
        )
    word_embedder()
    logger.info("END CIPHERTEXT EMBEDDING PHASE")
    return word_embedder.coocc_mat, word_embedder.sorted_voc


def unsupervised_translation():
    # python unsupervised.py --src_lang plain --tgt_lang cipher --src_emb data/plaintext.vec
    # --tgt_emb data/ciphertext.vec --n_refinement 5 --cuda False --emb_dim 50 --dis_most_frequent 0 --epoch_size 10000

    pass


def attack_enron(*args, **kwargs):
    setup_logger()
    # Params
    plaintext_voc_size = kwargs.get("plaintext_voc_size", 100)
    ciphertext_voc_size = kwargs.get("ciphertext_voc_size", 100)
    queryset_size = kwargs.get("queryset_size", 1000)
    logger.debug(f"Ciphertext vocabulary size: {ciphertext_voc_size}")
    logger.debug(f"Plaintext vocabulary size: {plaintext_voc_size}")
    if kwargs.get("L1"):
        logger.debug("L1 Scheme")
    else:
        logger.debug(f"L0 Scheme => Queryset size: {queryset_size}")

    logger.info("ATTACK BEGINS")

    similar_docs, stored_docs = split_df(df=extract_sent_mail_contents(), frac=0.4)

    plaintext_embedding_phase(similar_docs, plaintext_voc_size)
    ciphertext_embedding_phase(
        corpus_df=stored_docs,
        voc_size=ciphertext_voc_size,
        queryset_size=queryset_size,
        L1=kwargs.get("L1"),
    )

    unsupervised_translation()
    logger.warning("ATTACK ENDS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--plaintext-voc-size", type=int, default=100, help="Plaintext vocabulary size"
    )
    parser.add_argument(
        "--ciphertext-voc-size",
        type=int,
        default=100,
        help="Ciphertext vocabulary size",
    )
    parser.add_argument(
        "--queryset-size", type=int, default=1000, help="Fake queryset size"
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
    # TODO: same for iter number

    params = parser.parse_args()
    if (
        params.attack_dataset == "enron"
    ):  # We can use an enum or a dict if we have a lot of possible dataset
        attack_enron(**vars(params))
    else:
        raise ValueError("Unknown dataset")
