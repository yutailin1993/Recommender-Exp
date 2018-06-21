from sklearn.model_selection import train_test_split
import numpy as np


def gen_train_test(inputs, ratio=0.3):

    train_rating = np.zeros(
            (inputs.shape[0], inputs.shape[1]),
            dtype=np.int8)

    train_indices = []
    test_indices = []

    for usr in range(inputs.shape[0]):
        non_zero_indices = np.nonzero(inputs[usr])[0]
        non_zero_indices = np.random.permutation(non_zero_indices)
        
        train_idx = []
        test_idx = []
        for idx in range(int((1-ratio)*non_zero_indices.shape[0])):
            train_rating[usr][non_zero_indices[idx]] = 1
            train_idx.append(non_zero_indices[idx])

        for idx in range(idx+1, non_zero_indices.shape[0]):
            test_idx.append(non_zero_indices[idx])

        train_indices.append(train_idx)
        test_indices.append(test_idx)

    return train_rating, train_indices, test_indices


def hit_recall(topN, indices, N=5):
    """Calculate Recall

    -- Args --:
        topN: reconstruct top N recommand
        indices: list of items that user have seen in test set

    -- Return --:
        recall: most popular prdiction recall
    """

    topN_set = set(topN[:N])
    indice_set = set(indices[:N])

    hit_count = len(topN_set & indice_set)

    return hit_count / min(N, len(indice_set))


def recall_at_N(topN, indices, N=5):
    """Calculate Recall

    -- Args --:
        topN: reconstruct top N recommand
        indices: list of items that user have seen in test set

    -- Return --:
        recall: user's average precision
    """
    batch_recall_sum = 0.
    effect_sum = 0
    for i in range(topN.shape[0]):
        topN_set_i = set(topN[i][:N])
        indice_set_i = set(indices[i])
        hit_count = 0

        if len(indices[i]) != 0:
            hit_count = len(topN_set_i & indice_set_i)
            batch_recall_sum += hit_count / min(N, len(indice_set_i))
            effect_sum += 1

    try:
        return batch_recall_sum / effect_sum
    except ZeroDivisionError:
        return None


def avg_precision(topN, indices):
    '''
    Calculate Average Precision

    -- Args --:
        topN: reconstruct top N recommand
        indices: list of items that user have seen in test set

    -- Return --:
        ap: user's average precision
    '''
    N = topN.shape[1]
    batch_sum_p = 0.
    effect_sum = 0
    
    for i in range(topN.shape[0]):
        sum_p = 0.
        hit_count = 0
        for j in range(N):
            if topN[i][j] in indices[i]:
                hit_count += 1
                sum_p += hit_count / (j+1)
        try:
            batch_sum_p += sum_p/min(N, len(indices[i]))
            effect_sum += 1
        except ZeroDivisionError:
            continue

    try:
        return batch_sum_p / effect_sum
    except ZeroDivisionError:
        None


def get_topN(rec_matrix, train_index, N=5):

    topN = np.zeros((rec_matrix.shape[0], N), dtype=np.int32)
    recon_rank = rec_matrix.argsort()[:, ::-1]

    for i in range(rec_matrix.shape[0]):
        topN_idx = 0
        for rank_idx in recon_rank[i]:
            if topN_idx == N:
                break
            if rank_idx not in train_index[i]:
                topN[i][topN_idx] = rank_idx
                topN_idx += 1

    return topN
