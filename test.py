import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

to_iter = (
    list(assign.cipher_voc_info.keys())[-10:] + list(assign.cipher_voc_info.keys())[:10]
)
prediction = {}
for cipher_kw in tqdm.tqdm(
    iterable=to_iter, desc=f"Evaluating each plain-cipher pairs"
):
    cipher_ind = assign.cipher_voc_info[cipher_kw]["vector_ind"]
    cipher_prob = assign.cipher_voc_info[cipher_kw]["word_prob"]
    cipher_coocc = np.array(
        [
            assign.cipher_coocc[
                cipher_ind, assign.cipher_voc_info[known_cipher]["vector_ind"]
            ]
            for known_cipher in assign._known_queries.values()
        ]
    )
    score_list = []
    for plain_kw in assign.plain_voc_info.keys():
        plain_prob = assign.plain_voc_info[plain_kw]["word_prob"]
        plain_ind = assign.plain_voc_info[plain_kw]["vector_ind"]
        plain_coocc = np.array(
            [
                assign.plain_coocc[
                    plain_ind, assign.plain_voc_info[known_plain]["vector_ind"]
                ]
                for known_plain in assign._known_queries.values()
            ]
        )
        prob_diff = plain_prob - cipher_prob
        cocc_diff = plain_coocc - cipher_coocc
        instance = np.append(cocc_diff, prob_diff) ** 2
        score = 1 / np.max(instance[:-1]) + 1 / instance[-1]
        score_list.append((instance, score, plain_kw))
    score_list.sort(key=lambda tup: tup[1])
    prediction[cipher_kw] = score_list
