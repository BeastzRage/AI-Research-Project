import pandas as pd
from scipy.sparse import csr_matrix, coo_array
from recpack.util import get_top_K_ranks
from tqdm import tqdm

from src.metrics import calculate_ndcg, calculate_calibrated_recall
from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter


def matrix2df(X: csr_matrix) -> pd.DataFrame:
    """
    Coverts a sparse matrix into a dataframe.
    :param X: scipy sparse csr matrix
    :return: pandas.Dataframe represention of the sparse matrix
    """

    coo = coo_array(X)
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "value": coo.data
    })


def scores2recommendations(
    scores: csr_matrix,
    X_test_in: csr_matrix,
    recommendation_count: int,
    prevent_history_recos = True
) -> pd.DataFrame:
    """
    Coverts a score matrix containing item scores for each user into a dataframe containing the top k recommendations for each user.

    Provides the option to not include items the user has already interacted with in the final set of recommendations.

    :param scores: scipy sparse csr matrix containing item scores for each user
    :param X_test_in: scipy sparse csr matrix of the fold in test dataset
    :param recommendation_count: int, amount of recommendations to return for each user
    :param prevent_history_recos: boolean, when True does not include items the user has already interacted with in the recommendations
    :return: pandas.Dataframe containing the top specified amount of recommendations for each user.
    """


    # ensure you don't recommend fold-in items
    if prevent_history_recos:
        scores[(X_test_in > 0)] = 0
    # rank items
    ranks = get_top_K_ranks(scores, recommendation_count)
    # convert to a dataframe
    df_recos = matrix2df(ranks).rename(columns={"value": "rank"}).sort_values(["user_id", "rank"])
    return df_recos


def average_accuracy(interaction_matrix_csr, k):

    """
    Calculates the average NDCG@20 and Recall@20 scores of recommendations made using ItemKNN on a given set of interaction data.

    A strong generalization split of the given interaction data is made, leaving 80% of that data for training and the other 20% for testing.
    The test data is then split, 80% for a fold in set and 20% for the hold out set.

    :param interaction_matrix_csr: scipy csr matrix containing interaction data used for training and testing
    :param k: neighborhood size to use for ItemKNN
    :return: A tuple containing the average NDCG@20 and Recall@20 scores.
    """


    from src.ItemKNN import ItemKNN

    ndcg = 0
    recall = 0

    iterations = 100
    test_splitter = StrongGeneralizationSplitter()
    algo = ItemKNN(k)

    for i in tqdm(range(iterations), desc="calculating average NDCG@20 and Recall@20"):

        train, _, (test_fold_in, test_hold_out) = test_splitter.split(interaction_matrix_csr)

        # dataframe version of hold-out set to compute metrics
        df_test_out = matrix2df(test_hold_out)

        scores = algo.item_knn_scores(train, test_fold_in)
        df_recos = scores2recommendations(scores, test_fold_in, 20)

        ndcg = calculate_ndcg(df_recos, 20, df_test_out)
        recall = calculate_calibrated_recall(df_recos, 20, df_test_out)
        ndcg += ndcg
        recall += recall

    ndcg = ndcg / iterations
    recall = recall / iterations

    return ndcg, recall
