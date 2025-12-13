import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_array
from recpack.util import get_top_K_ranks
from recpack.matrix import InteractionMatrix
from recpack.scenarios import StrongGeneralization
from src.metrics import calculate_ndcg, calculate_calibrated_recall
from recpack.util import get_top_K_values

from src.SVD import SVD
from src.NCoreFilter import NCoreFilter
from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter
from src.ItemKNN import ItemKNN
from src.DataPreprocessor import DataPreprocessor

import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)


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


def main():


    user_reviews = pd.read_csv("dataset/user_reviews.csv")
    train_interactions = pd.read_csv("dataset/train_interactions.csv")

    # Merge train interactions into user reviews with value for 'recommend' = true if item user pair not present in user reviews
    train_interactions = train_interactions.assign(recommend=True)
    user_reviews = (
        pd.concat([user_reviews, train_interactions])
        .drop_duplicates(subset=["user_id", "item_id"], keep="first")
        .reset_index(drop=True))

    data_preprocessor = DataPreprocessor()
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.map_ids(user_reviews)
    interaction_matrix_csr = data_preprocessor.to_interaction_matrix(user_reviews)
    modified_interaction_matrix_csr = data_preprocessor.to_modified_interaction_matrix(user_reviews, -1)


    five_core_filter = NCoreFilter(5)
    interaction_matrix_csr, updated_item_id_mapping, updated_user_id_mapping = five_core_filter.filter(interaction_matrix_csr, new_to_old_item_id_mapping, new_to_old_user_id_mapping)
    modified_interaction_matrix_csr, updated_item_id_mapping_modified, updated_user_id_mapping_modified = five_core_filter.filter(modified_interaction_matrix_csr, new_to_old_item_id_mapping, new_to_old_user_id_mapping)


    validation_splitter = StrongGeneralizationSplitter(train_ratio=0.8, val_ratio=0.2, test_ratio=0, fold_in_ratio=0.8)

    validation_range = 100
    step_size = 1
    different_split_test_count = 5
    validation_scores: dict = {}

    best_k = None
    if best_k is None:
        for i in tqdm(range(different_split_test_count), desc="Optimizing hyper parameters"):
            train, (validation_fold_in, validation_hold_out), _, _ = validation_splitter.split(interaction_matrix_csr)
            k = 1
            for j in range(validation_range):
                # dataframe version of hold-out set to compute metrics
                df_validation_out = matrix2df(validation_hold_out)

                algo = ItemKNN()
                scores = algo.item_knn_scores(train, validation_fold_in, k)
                df_recos = scores2recommendations(scores, validation_fold_in, 20)

                ndcg = calculate_ndcg(df_recos, 20, df_validation_out)
                if k not in validation_scores or validation_scores[k] < ndcg:
                    validation_scores[k] = ndcg
                k += step_size

        best_k = max(validation_scores, key=validation_scores.get)
        print("Best baseline k: ", best_k)


    best_k_modified = None
    if best_k_modified is None:
        validation_scores: dict = {}
        for i in tqdm(range(different_split_test_count), desc="Optimizing hyper parameters"):
            train, (validation_fold_in, validation_hold_out), _, _ = validation_splitter.split(modified_interaction_matrix_csr)
            k = 1
            for j in range(validation_range):
                # dataframe version of hold-out set to compute metrics
                df_validation_out = matrix2df(validation_hold_out)

                algo = ItemKNN()
                scores = algo.item_knn_scores(train, validation_fold_in, k)
                df_recos = scores2recommendations(scores, validation_fold_in, 20)

                ndcg = calculate_ndcg(df_recos, 20, df_validation_out)
                if k not in validation_scores or validation_scores[k] < ndcg:
                    validation_scores[k] = ndcg
                k += step_size

        best_k_modified = max(validation_scores, key=validation_scores.get)
        print("Best modified k: ", best_k_modified)



    ndcgScores = {"base_itemKNN": 0.0, "mod_itemKNN": 0.0, "base_SVD": 0.0, "mod_SVD": 0.0}
    recallScores = {"base_itemKNN": 0.0, "mod_itemKNN": 0.0, "base_SVD": 0.0, "mod_SVD": 0.0}

    iterations = 100
    test_splitter = StrongGeneralizationSplitter()

    for i in tqdm(range(iterations), desc="calculating NDCG and recall for all scenarios: "):


        train, _, (test_fold_in, test_hold_out), _ = test_splitter.split(interaction_matrix_csr)

        # dataframe version of hold-out set to compute metrics
        df_test_out = matrix2df(test_hold_out)

        # Baseline ItemKNN
        algo = ItemKNN()
        scores = algo.item_knn_scores(train, test_fold_in, best_k)
        df_recos = scores2recommendations(scores, test_fold_in, 20)

        ndcg = calculate_ndcg(df_recos, 20, df_test_out)
        recall = calculate_calibrated_recall(df_recos, 20, df_test_out)
        ndcgScores["base_itemKNN"] += ndcg
        recallScores["base_itemKNN"] += recall


        # # Baseline SVD
        # mf = SVD(pd.DataFrame(train.toarray()))
        #
        # scores = csr_matrix(mf.get_scores(pd.DataFrame(test_fold_in.toarray())))
        #
        # df_recos = scores2recommendations(scores, test_fold_in, 20)
        #
        # ndcg = calculate_ndcg(df_recos, 20, df_test_out)
        # recall = calculate_calibrated_recall(df_recos, 20, df_test_out)
        # ndcgScores["base_SVD"] += ndcg
        # recallScores["base_SVD"] += recall
        #
        #
        #
        train, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = test_splitter.split(modified_interaction_matrix_csr)

        df_test_out = matrix2df(test_hold_out)

        #Modified interaction matrix ItemKNN
        scores = algo.item_knn_scores(train, test_fold_in, best_k_modified)
        df_recos = scores2recommendations(scores, test_fold_in, 20)

        ndcg = calculate_ndcg(df_recos, 20, df_test_out)
        recall = calculate_calibrated_recall(df_recos, 20, df_test_out)
        ndcgScores["mod_itemKNN"] += ndcg
        recallScores["mod_itemKNN"] += recall

        #
        # #Modified interaction matrix SVD
        # mf = SVD(pd.DataFrame(train.toarray()))
        #
        # scores = csr_matrix(mf.get_scores(pd.DataFrame(test_fold_in.toarray())))
        #
        # df_recos = scores2recommendations(scores, test_fold_in, 10)
        #
        # ndcg = calculate_ndcg(df_recos, 10, df_test_out)
        # recall = calculate_calibrated_recall(df_recos, 10, df_test_out)
        # ndcgScores["mod_SVD"] += ndcg
        # recallScores["mod_SVD"] += recall

    for key in ndcgScores:
        ndcgScores[key] = ndcgScores[key] / iterations
        recallScores[key] = recallScores[key] / iterations

        print(f"{key}:")
        print(f"  NDCG@10: {ndcgScores[key]:.5f}")
        print(f"Recall@10: {recallScores[key]:.5f}\n\n")

    test_reviews = pd.read_csv("dataset/test_interactions_in.csv")

    # Filter out items not seen in the training set
    # (ItemKNN cannot score unseen items)
    old_to_new_item_id_mapping = {v: k for k, v in updated_item_id_mapping.items()}
    test_reviews = test_reviews[test_reviews['item_id'].isin(old_to_new_item_id_mapping)]

    # Remap user ids
    user_ids = test_reviews['user_id'].unique()
    user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
    new_to_old_test_user_id_mapping = {v: k for k, v in user_id_mapping.items()}
    test_reviews['user_id'] = test_reviews['user_id'].map(user_id_mapping)

    # update item ids to correspond to that of the training set
    test_reviews['item_id'] = test_reviews['item_id'].map(old_to_new_item_id_mapping)

    # Build CSR matrix with full train-set dimensions
    test_rows = test_reviews['user_id'].values
    test_cols = test_reviews['item_id'].values
    test_data = np.ones(len(test_reviews), dtype=np.int8)

    test_fold_in_matrix_csr = csr_matrix(
        (test_data, (test_rows, test_cols)),
        shape=(len(test_reviews['user_id'].unique()), interaction_matrix_csr.shape[1])  # <— same as training matrix
    )



    # Baseline ItemKNN
    algo = ItemKNN()
    scores = algo.item_knn_scores(interaction_matrix_csr, test_fold_in_matrix_csr, best_k)
    df_recos = scores2recommendations(scores, test_fold_in_matrix_csr, 20)


    df_recos = df_recos.drop('rank', axis=1)

    for id in df_recos['user_id'].unique().tolist():
        if id not in new_to_old_test_user_id_mapping:
            raise ValueError("User id not found in user-id mapping")
    for id in df_recos['item_id'].unique().tolist():
        if id not in updated_item_id_mapping:
            raise ValueError("Item id not found in item id mapping")

    df_recos['user_id'] = df_recos['user_id'].replace(new_to_old_test_user_id_mapping)
    df_recos['item_id'] = df_recos['item_id'].replace(updated_item_id_mapping)
    df_recos.to_csv("recommendation_results/baseline_recommendations.csv", index=False)


    # modified interaction matrix ItemKNN
    algo = ItemKNN()
    scores = algo.item_knn_scores(modified_interaction_matrix_csr, test_fold_in_matrix_csr, best_k_modified)
    df_recos = scores2recommendations(scores, test_fold_in_matrix_csr, 20)


    df_recos = df_recos.drop('rank', axis=1)
    for id in df_recos['user_id'].unique().tolist():
        if id not in new_to_old_test_user_id_mapping:
            raise ValueError("User id not found in user-id mapping")
    for id in df_recos['item_id'].unique().tolist():
        if id not in updated_item_id_mapping_modified:
            raise ValueError("Item id not found in item id mapping")

    df_recos['user_id'] = df_recos['user_id'].replace(new_to_old_test_user_id_mapping)
    df_recos['item_id'] = df_recos['item_id'].replace(updated_item_id_mapping_modified)
    df_recos.to_csv("recommendation_results/modified_matrix_recommendations.csv", index=False)



    # print("iteration 0")
    # i = 0
    # while True:
    #     i += 1
    #     print(f"\riteration {i}")
    #
    #     train, _, (test_fold_in, test_hold_out), _ = test_splitter.split(interaction_matrix_csr)
    #
    #     # dataframe version of hold-out set to compute metrics
    #     df_test_out = matrix2df(test_hold_out)
    #
    #     # Baseline ItemKNN
    #     algo = ItemKNN()
    #     scores = algo.item_knn_scores(train, test_fold_in, best_k)
    #     df_recos = scores2recommendations(scores, test_fold_in, 20)
    #
    #     ndcg = calculate_ndcg(df_recos, 20, df_test_out)
    #
    #
    #     if abs(ndcgScores["base_itemKNN"] - ndcg) < 0.0001:
    #         df_recos = df_recos.drop('rank', axis=1)
    #         for id in df_recos['user_id'].unique().tolist():
    #             if id not in updated_user_id_mapping:
    #                 raise ValueError("User id not found in user-id mapping")
    #         for id in df_recos['item_id'].unique().tolist():
    #             if id not in updated_item_id_mapping:
    #                 raise ValueError("Item id not found in item id mapping")
    #
    #         df_recos['user_id'] = df_recos['user_id'].replace(updated_user_id_mapping)
    #         df_recos['item_id'] = df_recos['item_id'].replace(updated_item_id_mapping)
    #         df_recos.to_csv("recommendations.csv", index=False)
    #         break





    return 0

if __name__ == "__main__":
    main()