import logging

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
        vector_filename="plain_vector",
        occ_array=extractor.occ_array,
        voc_with_occ=extractor.sorted_voc,
    )
    word_embedder()
    logger.info("END PLAINTEXT EMBEDDING PHASE")
    return word_embedder.coocc_mat, word_embedder.sorted_voc


def ciphertext_embedding_phase(
    corpus_df, voc_size=100, minimum_freq=1, queryset_size=1000
):
    logger.info("START CIPHERTEXT EMBEDDING PHASE")
    extractor = QueryResultExtractor(corpus_df, voc_size, minimum_freq)

    word_embedder = GloveWordEmbedding(
        vocab_filename="cipher_voc.txt",
        vector_filename="cipher_vector",
        query_ans_dict=extractor.get_query_answer(size=queryset_size),
    )
    word_embedder()
    logger.info("END CIPHERTEXT EMBEDDING PHASE")
    return word_embedder.coocc_mat, word_embedder.sorted_voc


def unsupervised_translation():
    pass


def attack_enron(plain_voc_size=100, cipher_voc_size=100, queryset_size=1000):
    setup_logger()
    logger.info("ATTACK BEGINS")

    similar_docs, stored_docs = split_df(df=extract_sent_mail_contents(), frac=0.4)

    plaintext_embedding_phase(similar_docs, plain_voc_size)
    ciphertext_embedding_phase(
        corpus_df=stored_docs, voc_size=cipher_voc_size, queryset_size=queryset_size
    )

    unsupervised_translation()
    logger.warning("ATTACK ENDS")


if __name__ == "__main__":
    attack_enron()
