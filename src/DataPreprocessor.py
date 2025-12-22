import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataPreprocessor:

    def map_ids(self, interaction_data: pd.DataFrame):

        interaction_data.rename(columns={'user_id': 'old_user_id', 'item_id': 'old_item_id'}, inplace=True)

        user_ids = interaction_data['old_user_id'].unique()
        user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(user_ids)}

        interaction_data['user_id'] = interaction_data['old_user_id'].map(user_id_mapping)

        item_ids = interaction_data['old_item_id'].unique()
        item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(item_ids)}

        interaction_data['item_id'] = interaction_data['old_item_id'].map(item_id_mapping)

        new_to_old_user_id_mapping = {v: k for k, v in user_id_mapping.items()}
        new_to_old_item_id_mapping = {v: k for k, v in item_id_mapping.items()}

        return new_to_old_user_id_mapping, new_to_old_item_id_mapping


    def to_interaction_matrix(self, interaction_data: pd.DataFrame) -> csr_matrix:
        interaction_data = interaction_data.drop_duplicates(subset=['user_id', 'item_id'])
        num_users = len(interaction_data['old_user_id'].unique())
        num_items = len(interaction_data['old_item_id'].unique())

        rows = interaction_data['user_id'].values
        cols = interaction_data['item_id'].values

        data = np.ones(len(interaction_data), dtype=np.int8)
        interaction_matrix_csr = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
        return interaction_matrix_csr

    def to_modified_interaction_matrix(self, interaction_data: pd.DataFrame, negative_interaction_value: float=-1) -> csr_matrix:
        assert 'recommend' in interaction_data.columns, "dataframe must contain 'recommend' column"

        interaction_data = interaction_data.drop_duplicates(subset=['user_id', 'item_id'])
        num_users = len(interaction_data['old_user_id'].unique())
        num_items = len(interaction_data['old_item_id'].unique())

        rows = interaction_data['user_id'].values
        cols = interaction_data['item_id'].values

        data = np.where(interaction_data['recommend'].values, 1, negative_interaction_value).astype(np.int8)
        modified_interaction_matrix_csr = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
        return modified_interaction_matrix_csr


    def to_test_fold_in_matrix(self, test_reviews: pd.DataFrame, train_item_id_mapping: dict[int, int]):

        # Filter out items not seen in the training set
        # (ItemKNN cannot score unseen items)
        old_to_new_item_id_mapping = {v: k for k, v in train_item_id_mapping.items()}
        test_reviews = test_reviews[test_reviews['item_id'].isin(old_to_new_item_id_mapping)].copy()

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
            shape=(len(test_reviews['user_id'].unique()), len(train_item_id_mapping))  # <— same as training matrix
        )

        return test_fold_in_matrix_csr, new_to_old_test_user_id_mapping