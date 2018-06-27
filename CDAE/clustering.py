import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import operator
import tensorflow as tf
from CDAE import AutoEncoder


def count_score(top_lists, label_count, total_usr, exponent=1.0001, alpha=100):
    """
    Use Zipf's distribution to calculate score.
    """
    score_map = {}
    N = len(top_lists[0])

    constant = 0
    for k in range(N):
        constant += 1 / (k+1)**exponent

    for list_id, i in enumerate(top_lists):
        for idx, j in enumerate(i):
            zipf_score = 1 / (idx+1)**exponent / constant
            if j not in score_map:
                score_map[j] = alpha * zipf_score * (label_count[list_id] / total_usr)
            else:
                score_map[j] += alpha * zipf_score * (label_count[list_id] / total_usr)

    return score_map


def get_pca_out(inputs, pcaModel=None):
    if pcaModel is None:
        pca = PCA(n_components=10, svd_solver='full')
    else:
        pca = pcaModel

    pca_out = pca.fit_transform(inputs)

    return pca_out


def calculate_kmeans(inputs, kmeansModel=None, NUM_CLUSTER=10):
    if kmeansModel is None:
        kmeans = KMeans(n_clusters=NUM_CLUSTER, n_init=10, algorithm='full')
    else:
        kmeans = kmeansModel

    kmeans.fit(inputs)

    return kmeans


def get_cluster_attributes(cluster_model, NUM_CLUSTER=10):
    label_count = {}
    for i in cluster_model.labels_:
        if i not in label_count:
            label_count[i] = 1
        else:
            label_count[i] += 1

    label_index = {}
    for i in range(NUM_CLUSTER):
        label_index[i] = []

    label_list = list(cluster_model.labels_)

    for idx, i in enumerate(label_list):
        label_index[i].append(idx)

    return label_index, label_count


def calculate_cluster_top(allData, total_usr, total_item,
        NUM_CLUSTER=10, batch_size=1, weight=1, denoising=True):
    cluster_top = []

    for c in range(NUM_CLUSTER):
        train_user_c = allData['LABEL_INDEX'][c]
        test_user_c = [x for x in allData['LABEL_INDEX'][c] if x in allData['TEST_USER_NOW']]

        if len(test_user_c) == 0:
            continue

        train_matrix_c = np.take(allData['TRAIN_MATRIX'], train_user_c, axis=0)
        test_matrix_c = np.take(allData['TEST_MATRIX_NOW'], test_user_c, axis=0)

        if weight != 1:
            top_n = np.count_nonzero(train_matrix_c, axis=0).argsort()[::-1][:30]
            with_weight = True
        else:
            top_n = None
            with_weight = False

        tf.reset_default_graph()

        autoEncoder = AutoEncoder(
                user_num=total_usr,
                item_num=total_item,
                mode='user',
                loss_function='log_loss',
                with_weight=with_weight,
                denoising=denoising,
                batch_size=batch_size,
                epochs=100)

        autoEncoder.model_load(1)

        autoEncoder.train_all(rating=train_matrix_c, train_idents=train_user_c, topN=top_n, weight=weight)

        test_out = autoEncoder.predict(test_matrix_c, test_user_c)

        test_out_upper_quartile = []
        upper_quartile = np.percentile(test_out, 75, axis=0)
        for i in range(test_out.shape[1]):
            test_out_upper_quartile.append(
                    np.mean([x for x in test_out.T[i] if x > upper_quartile[i]]))

        test_out_upper_quartile = np.asarray(test_out_upper_quartile)

        rank_upper_quartile = test_out_upper_quartile.argsort()[::-1][:1000]

        cluster_top.append(rank_upper_quartile)

    return cluster_top


def get_score_top(score_map, N=10):
    sorted_rank = sorted(score_map.items(), key=operator.itemgetter(1), reverse=True)
    sorted_rank = np.asarray(sorted_rank)

    top_N = np.asarray(sorted_rank[:, 0][:N], dtype=np.int32)

    return top_N
