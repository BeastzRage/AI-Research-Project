import numpy as np
import pandas as pd
import scipy as sp
from numpy.lib.npyio import savetxt
from scipy.sparse import csr_matrix
from recpack.util import get_top_K_ranks
from recpack.matrix import InteractionMatrix
from recpack.scenarios import StrongGeneralization
from metrics import calculate_ndcg, calculate_calibrated_recall
import warnings


user_reviews = pd.read_csv("dataset/user_reviews.csv")
user_reviews.head(20)
user_reviews.rename(columns={'user_id': 'old_user_id', 'item_id': 'old_item_id'}, inplace=True)

user_ids = user_reviews['old_user_id'].unique()
user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(user_ids)}

user_reviews['user_id'] = user_reviews['old_user_id'].map(user_id_mapping)

item_ids = user_reviews['old_item_id'].unique()
item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(item_ids)}

user_reviews['item_id'] = user_reviews['old_item_id'].map(item_id_mapping)

new_to_old_user_id_mapping = {v: k for k, v in user_id_mapping.items()}
new_to_old_item_id_mapping = {v: k for k, v in item_id_mapping.items()}


num_users = len(user_reviews['old_user_id'].unique())
num_items = len(user_reviews['old_item_id'].unique())

rows = user_reviews['user_id'].values
cols = user_reviews['item_id'].values

data = np.ones(len(user_reviews), dtype=np.int8)
interaction_matrix_csr = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

data = np.where(user_reviews['recommend'].values, 1, -1).astype(np.int8)
modified_interaction_matrix_csr = sp.sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))


# noinspection PyUnresolvedReferences
def my_cosine_similarity(X: sp.sparse.csr_matrix) -> sp.sparse.csr_matrix:
    # TODO: implement cosine similarity function
    # hint: should return a sparse (item x item)-matrix
    # hint: work with the entire matrix to benefit from highly optimized vectorized operations in numpy and scipy
    # important: don't forget to set the diagonal to 0 (no self-similarity)

    # a.b
    num = X.T @ X

    # ||a||2.||b||2
    norms = np.sqrt(np.array(X.power(2).sum(axis=0))).ravel()
    denom = np.outer(norms, norms)

    # dont devide by 0
    denom[denom == 0] = 1e-10

    # cos(a,b)
    cos_sim = num.multiply(1 / denom)

    # set diagonal to 0
    cos_sim.setdiag(0)
    cos_sim.eliminate_zeros()

    return cos_sim.tocsr()

# helper function to turn a sparse matrix into a dataframe
def matrix2df(X) -> pd.DataFrame:
    coo = sp.sparse.coo_array(X)
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "value": coo.data
    })

# helper function to convert a score matrix into a dataframe of recommendations
def scores2recommendations(
    scores: sp.sparse.csr_matrix,
    X_test_in: sp.sparse.csr_matrix,
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


# create a Recpack interaction matrix
X = InteractionMatrix.from_csr_matrix(interaction_matrix_csr)

# split matrix using strong generalization
# we use a fixed seed for reproducibility
scenario = StrongGeneralization(frac_users_train=0.8, frac_interactions_in=0.8, validation=False, seed=42)
scenario.split(X)

# get interaction matrices
X_test_in = scenario.test_data_in.values
X_test_out = scenario.test_data_out.values

# dataframe version of hold-out set to compute metrics later on
df_test_out = matrix2df(X_test_out)


from recpack.util import get_top_K_values

def item_knn_scores(
    X_train: sp.sparse.csr_matrix,
    X_test_in: sp.sparse.csr_matrix,
    neighbor_count: int
) -> sp.sparse.csr_matrix:
    # TODO: add code to compute scores for all (user, item) pairs
    # hint: use your own cosine similarity function
    # hint: use `get_top_K_values` to prune the similarity matrix
    # hint: think about how you can calculate the scores for all pairs at once

    cos_sim = my_cosine_similarity(X_train)
    cos_sim = get_top_K_values(cos_sim, neighbor_count)

    # H . S
    scores = X_test_in @ cos_sim

    return scores.tocsr()

scores = item_knn_scores(interaction_matrix_csr, X_test_in, 50)
df_recos = scores2recommendations(scores, X_test_in, 10)

ndcg = calculate_ndcg(df_recos, 10, df_test_out)
recall = calculate_calibrated_recall(df_recos, 10, df_test_out)


print(df_recos.head(10))
print(f"  NDCG@10: {ndcg:.5f}")
print(f"Recall@10: {recall:.5f}")

scores = item_knn_scores(modified_interaction_matrix_csr, X_test_in, 50)
df_recos = scores2recommendations(scores, X_test_in, 10)

ndcg = calculate_ndcg(df_recos, 10, df_test_out)
recall = calculate_calibrated_recall(df_recos, 10, df_test_out)


print(df_recos.head(10))
print(f"  NDCG@10: {ndcg:.5f}")
print(f"Recall@10: {recall:.5f}")