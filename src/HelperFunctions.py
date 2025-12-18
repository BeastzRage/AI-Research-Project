import pandas as pd
from scipy.sparse import csr_matrix, coo_array
from recpack.util import get_top_K_ranks
from tqdm import tqdm

from src.metrics import calculate_ndcg, calculate_calibrated_recall
from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter


# helper function to turn a sparse matrix into a dataframe
def matrix2df(X) -> pd.DataFrame:
    coo = coo_array(X)
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "value": coo.data
    })

# helper function to convert a score matrix into a dataframe of recommendations
def scores2recommendations(
    scores: csr_matrix,
    X_test_in: csr_matrix,
    recommendation_count: int,
    prevent_history_recos = True
) -> pd.DataFrame:
    # ensure you don't recommend fold-in items
    if prevent_history_recos:
        scores[(X_test_in > 0)] = 0
    # rank items
    ranks = get_top_K_ranks(scores, recommendation_count)
    # convert to a dataframe
    df_recos = matrix2df(ranks).rename(columns={"value": "rank"}).sort_values(["user_id", "rank"])
    return df_recos


def average_accuracy(interaction_matrix_csr, best_k):

    from src.ItemKNN import ItemKNN

    ndcg = 0
    recall = 0

    iterations = 100
    test_splitter = StrongGeneralizationSplitter()
    algo = ItemKNN(best_k)

    for i in tqdm(range(iterations), desc="calculating average NDCG@20 and Recall@20"):

        train, _, (test_fold_in, test_hold_out), _ = test_splitter.split(interaction_matrix_csr)

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
