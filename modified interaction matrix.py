import numpy as np
import pandas as pd
import scipy as sp
from numpy.lib.npyio import savetxt
from scipy.sparse import csr_matrix, lil_matrix, coo_array, coo_matrix
from recpack.util import get_top_K_ranks
from recpack.matrix import InteractionMatrix
from recpack.scenarios import StrongGeneralization
from metrics import calculate_ndcg, calculate_calibrated_recall
from tqdm import tqdm
import warnings

from SVD import SVD

# noinspection PyUnresolvedReferences
def my_cosine_similarity(X: csr_matrix) -> csr_matrix:
    # returns a sparse (item x item)-matrix
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

def MinUsersPerItem(interaction_matrix_csr, item_mapping, min_users):
    # Filter out items with strictly less than min_users interactions
    item_mask = interaction_matrix_csr.getnnz(axis=0) >= min_users
    # Hint: use the .getnnz() method of the scipy csr matrix.
    # Apply the mask to the matrix to filter items
    filtered_interaction_matrix = interaction_matrix_csr[:, item_mask]

    # Great, because we're removing items (columns) from the matrix, we break our original mapping between columns and "old" item IDs!
    # Update the mapping:
    item_ids = np.array(list(item_mapping.keys()))
    good_item_ids = item_ids[item_mask]
    updated_item_mapping = {old_id: new_id for new_id, old_id in enumerate(good_item_ids)}

    return filtered_interaction_matrix, updated_item_mapping


def MinItemsPerUser(interaction_matrix_csr, user_mapping, min_items):
    # Filter out users with strictly less than min_items interactions
    user_mask = interaction_matrix_csr.getnnz(axis=1) >= min_items
    # Hint: use the .getnnz() method of the scipy csr matrix.

    # Apply the mask to the matrix to filter users
    filtered_interaction_matrix = interaction_matrix_csr[user_mask]

    # Great, because we're removing users (rows) from the matrix, we break our original mapping between rows and "old" user IDs!
    # Update the mapping:

    user_ids = np.array(list(user_mapping.keys()))
    good_user_ids = user_ids[user_mask]
    updated_user_mapping = {old_id: new_id for new_id, old_id in enumerate(good_user_ids)}

    return filtered_interaction_matrix, updated_user_mapping

class StrongGeneralizationSplitter:
    def __init__(self, train_ratio=0.8, val_ratio=0, test_ratio=0.2, fold_in_ratio=0.8):
        assert train_ratio + val_ratio + test_ratio == 1, "Train, validation, and test ratios must sum to 1."
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.fold_in_ratio = fold_in_ratio
        self.hold_out_ratio = 1 - fold_in_ratio

    def split(self, interaction_matrix_csr):
        num_users, num_items = interaction_matrix_csr.shape

        # Shuffle users for random splitting
        users = np.arange(num_users)
        np.random.shuffle(users)

        # Split users into training, validation, and test sets based on the specified ratios
        train_end = int(num_users * self.train_ratio)
        val_end = train_end + int(num_users * self.val_ratio)
        # Hint: this is where you'd use your train and validation ratios.

        train_users = users[:train_end]
        val_users = users[train_end:val_end]
        test_users = test_users = users[val_end:]
        # Hint: this is where you should slice the "users"

        train_matrix = interaction_matrix_csr[train_users, :]
        val_matrix = interaction_matrix_csr[val_users, :]
        test_matrix = interaction_matrix_csr[test_users, :]

        # Create fold-in and hold-out splits for validation and test sets
        val_fold_in, val_hold_out = self.split_interactions(val_matrix)
        test_fold_in, test_hold_out = self.split_interactions(test_matrix)

        return train_matrix, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users,
                                                                                          test_users)

    def split_interactions(self, interaction_matrix_csr):
        # Convert the matrix to lil format for easier manipulation
        lil = interaction_matrix_csr.tolil()

        # Initialize the lists to store fold-in and hold-out data
        fold_in_lil = lil_matrix((interaction_matrix_csr.shape[0], interaction_matrix_csr.shape[1]))
        hold_out_lil = lil_matrix((interaction_matrix_csr.shape[0], interaction_matrix_csr.shape[1]))

        # Iterate over each row (user)
        for i in tqdm(range(lil.shape[0]), desc="Splitting interactions: "):
            # Get the non-zero entries in this row
            row_data = lil.data[i]
            row_indices = lil.rows[i]

            # Determine the number of interactions to include in fold-in
            fold_in_size = int(len(row_data) * self.fold_in_ratio)

            # Randomly shuffle the indices
            shuffle_indices = np.random.permutation(len(row_data))

            # Split the indices into fold-in and fold-out
            fold_in_indices = shuffle_indices[:fold_in_size]
            hold_out_indices = shuffle_indices[fold_in_size:]

            # Assign the fold-in data to the fold_in_lil matrix
            fold_in_lil.rows[i] = np.array(row_indices)[fold_in_indices].tolist()
            fold_in_lil.data[i] = np.array(row_data)[fold_in_indices].tolist()

            # Assign the fold-out data to the hold_out_lil matrix
            hold_out_lil.rows[i] = np.array(row_indices)[hold_out_indices].tolist()
            hold_out_lil.data[i] = np.array(row_data)[hold_out_indices].tolist()

        # Convert the lil matrices back to csr format
        fold_in_matrix = fold_in_lil.tocsr()
        hold_out_matrix = hold_out_lil.tocsr()

        return fold_in_matrix, hold_out_matrix






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
modified_interaction_matrix_csr = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))




from recpack.util import get_top_K_values

def item_knn_scores(
    X_train: csr_matrix,
    X_test_in: csr_matrix,
    neighbor_count: int
) -> csr_matrix:
    # TODO: add code to compute scores for all (user, item) pairs
    # hint: use your own cosine similarity function
    # hint: use `get_top_K_values` to prune the similarity matrix
    # hint: think about how you can calculate the scores for all pairs at once

    cos_sim = my_cosine_similarity(X_train)
    cos_sim = get_top_K_values(cos_sim, neighbor_count)

    # H . S
    scores = X_test_in @ cos_sim

    return scores.tocsr()


interaction_matrix_csr, updated_user_id_mapping = MinItemsPerUser(interaction_matrix_csr, new_to_old_user_id_mapping, 5)
interaction_matrix_csr, updated_item_id_mapping = MinUsersPerItem(interaction_matrix_csr, new_to_old_item_id_mapping, 5)
modified_interaction_matrix_csr, updated_user_id_mapping_modified = MinItemsPerUser(modified_interaction_matrix_csr, new_to_old_user_id_mapping, 5)
modified_interaction_matrix_csr, updated_item_id_mapping_modified = MinUsersPerItem(modified_interaction_matrix_csr, new_to_old_item_id_mapping, 5)

splitter = StrongGeneralizationSplitter()
ndcg1 = 0
recall1 = 0
ndcg2 = 0
recall2 = 0
ndcg3 = 0
recall3 = 0
ndcg4 = 0
recall4 = 0
ndcg5 = 0
recall5 = 0
iterations = 100
for i in range(iterations):

    train, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = splitter.split(interaction_matrix_csr)


    scores = item_knn_scores(train, test_fold_in, 50)
    df_recos = scores2recommendations(scores, test_fold_in, 10)

    # dataframe version of hold-out set to compute metrics
    df_test_out = matrix2df(test_hold_out)

    ndcg = calculate_ndcg(df_recos, 10, df_test_out)
    recall = calculate_calibrated_recall(df_recos, 10, df_test_out)
    ndcg1 += ndcg
    recall1 += recall

    # print(df_recos.head(10))
    print(f"  NDCG@10: {ndcg:.5f}")
    print(f"Recall@10: {recall:.5f}")

    train, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = splitter.split(modified_interaction_matrix_csr)

    df_test_out = matrix2df(test_hold_out)

    scores = item_knn_scores(train, test_fold_in, 50)
    df_recos = scores2recommendations(scores, test_fold_in, 10)

    ndcg = calculate_ndcg(df_recos, 10, df_test_out)
    recall = calculate_calibrated_recall(df_recos, 10, df_test_out)
    ndcg2 += ndcg
    recall2 += recall

    # print(df_recos.head(10))
    print(f"  NDCG@10: {ndcg:.5f}")
    print(f"Recall@10: {recall:.5f}")

    # create a Recpack interaction matrix
    X = InteractionMatrix.from_csr_matrix(interaction_matrix_csr)

    # split matrix using strong generalization
    # we use a fixed seed for reproducibility
    scenario = StrongGeneralization(frac_users_train=0.8, frac_interactions_in=0.8, validation=False)
    scenario.split(X)

    # get interaction matrices
    X_train = scenario.full_training_data.values
    X_test_in = scenario.test_data_in.values
    X_test_out = scenario.test_data_out.values

    # dataframe version of hold-out set to compute metrics later on
    df_test_out = matrix2df(X_test_out)

    scores = item_knn_scores(X_train, X_test_in, 50)
    df_recos = scores2recommendations(scores, X_test_in, 10)

    ndcg = calculate_ndcg(df_recos, 10, df_test_out)
    recall = calculate_calibrated_recall(df_recos, 10, df_test_out)
    ndcg3 += ndcg
    recall3 += recall

    # print(df_recos.head(10))
    print(f"  NDCG@10: {ndcg:.5f}")
    print(f"Recall@10: {recall:.5f}\n\n")


    train, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = splitter.split(interaction_matrix_csr)

    df_test_out = matrix2df(test_hold_out)

    mf = SVD(pd.DataFrame(train.toarray()))

    scores = csr_matrix(mf.get_scores(pd.DataFrame(test_fold_in.toarray())))

    df_recos = scores2recommendations(scores, test_fold_in, 10)

    ndcg = calculate_ndcg(df_recos, 10, df_test_out)
    recall = calculate_calibrated_recall(df_recos, 10, df_test_out)
    ndcg4 += ndcg
    recall4 += recall

    # print(df_recos.head(10))
    print(f"  NDCG@10: {ndcg:.5f}")
    print(f"Recall@10: {recall:.5f}")


    train, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = splitter.split(modified_interaction_matrix_csr)

    df_test_out = matrix2df(test_hold_out)

    mf = SVD(pd.DataFrame(train.toarray()))

    scores = csr_matrix(mf.get_scores(pd.DataFrame(test_fold_in.toarray())))

    df_recos = scores2recommendations(scores, test_fold_in, 10)

    ndcg = calculate_ndcg(df_recos, 10, df_test_out)
    recall = calculate_calibrated_recall(df_recos, 10, df_test_out)
    ndcg5 += ndcg
    recall5 += recall

    # print(df_recos.head(10))
    print(f"  NDCG@10: {ndcg:.5f}")
    print(f"Recall@10: {recall:.5f}")



ndcg1 /= iterations
recall1 /= iterations
ndcg2 /= iterations
recall2 /= iterations
ndcg3 /= iterations
recall3 /= iterations
ndcg4 /= iterations
recall4 /= iterations
ndcg5 /= iterations
recall5 /= iterations

print(f"RESULTS:\n")

print("Normal ItemKNN")
print(f"  NDCG@10: {ndcg1:.5f}")
print(f"Recall@10: {recall1:.5f}\n\n")
print("Modified ItemKNN")
print(f"  NDCG@10: {ndcg2:.5f}")
print(f"Recall@10: {recall2:.5f}\n\n")
print("Recpack ItemKNN")
print(f"  NDCG@10: {ndcg3:.5f}")
print(f"Recall@10: {recall3:.5f}\n\n")
print("Normal SVD")
print(f"  NDCG@10: {ndcg4:.5f}")
print(f"Recall@10: {recall4:.5f}\n\n")
print("Modified SVD")
print(f"  NDCG@10: {ndcg5:.5f}")
print(f"Recall@10: {recall5:.5f}\n\n")