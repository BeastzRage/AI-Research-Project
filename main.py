import pandas as pd

from src.NCoreFilter import NCoreFilter
from src.ItemKNN import ItemKNN
from src.DataPreprocessor import DataPreprocessor
from src.HelperFunctions import matrix2df, scores2recommendations, average_accuracy

import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)


def main():


    print("Loading training data...", end='\r')
    user_reviews = pd.read_csv("dataset/user_reviews.csv")
    print("Loading training data... [DONE]")

    print("Preprocessing data...", end='\r')
    negative_interaction_value = -1
    data_preprocessor = DataPreprocessor()
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.map_ids(user_reviews)
    interaction_matrix_csr = data_preprocessor.to_modified_interaction_matrix(user_reviews, negative_interaction_value)
    print("Preprocessing data... [DONE]")

    print("Filtering data...", end='\r')
    two_core_filter = NCoreFilter(2)
    interaction_matrix_csr, updated_item_id_mapping, updated_user_id_mapping = two_core_filter.filter(
        interaction_matrix_csr, new_to_old_item_id_mapping, new_to_old_user_id_mapping)
    print("Filtering data... [DONE]")

    print("Loading testing data...", end='\r')
    test_reviews = pd.read_csv("dataset/test_interactions_in.csv")

    test_fold_in_matrix_csr, new_to_old_test_user_id_mapping = data_preprocessor.to_test_fold_in_matrix(test_reviews, updated_item_id_mapping)

    print("Loading testing data... [DONE]")

    print("Generating recommendations...")

    # generate recommendations using ItemKNN
    algo = ItemKNN()
    best_k = algo.neighborhood_tuning(interaction_matrix_csr, 0.8, 0.8, 100, 1, 5)
    scores = algo.item_knn_scores(interaction_matrix_csr, test_fold_in_matrix_csr)
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
    df_recos.to_csv(f"recommendation_results/modified_matrix_{negative_interaction_value}.csv", index=False)

    print("generating recommendations... [DONE]")

    # print("Running offline accuracy test...", end='\r')
    #
    # ndcg, recall = average_accuracy(interaction_matrix_csr, best_k)
    # print("Average accuracy score:")
    # print(f"  NDCG@20: {ndcg:.5f}")
    # print(f"Recall@20: {recall:.5f}\n\n")
    #
    # print("Running offline accuracy test... [DONE]")

if __name__ == "__main__":
    main()