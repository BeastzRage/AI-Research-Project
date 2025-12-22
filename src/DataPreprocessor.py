import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataPreprocessor:

    def map_ids(self, interaction_data: pd.DataFrame):
        """
        Remaps user and item IDs to consecutive integers starting from 0 so they can be used in a sparse matrix.

        New columns are added to the dataframe named 'old_user_id' and 'old_item_id' to store the old user and item IDs
        respectively. The old IDs in the 'user_id' and 'item_id' columns are replaced with their newly mapped IDs.

        :param interaction_data: pandas.DataFrame containing user-item interactions. Must contain 'user_id' and 'item_id' columns.
        :return: tuple(new_to_old_user_id_mapping, new_to_old_item_id_mapping)
            dict new_to_old_user_id_mapping maps the newly assigned user IDs to old user IDs
            dict new_to_old_item_id_mapping maps the newly assigned item IDs to old item IDs
        """

        assert 'user_id' in interaction_data.columns, "dataframe must contain 'user_id' column"
        assert 'item_id' in interaction_data.columns, "dataframe must contain 'item_id' column"

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
        """
        Converts a pandas.DataFrame containing user-item interactions to a scipy sparse csr matrix.

        User and item IDs should be remapped prior to obtain a meaningful matrix.
        Duplicate interaction are dropped prior to converting to sparse matrix.

        :param interaction_data: pandas.DataFrame containing user-item interactions. Must contain 'user_id' and 'item_id' columns.
        :return: scipy.sparse.csr_matrix of the interaction data.
        """


        assert 'user_id' in interaction_data.columns, "dataframe must contain 'user_id' column"
        assert 'item_id' in interaction_data.columns, "dataframe must contain 'item_id' column"
        interaction_data = interaction_data.drop_duplicates(subset=['user_id', 'item_id'])
        num_users = len(interaction_data['user_id'].unique())
        num_items = len(interaction_data['item_id'].unique())

        rows = interaction_data['user_id'].values
        cols = interaction_data['item_id'].values

        data = np.ones(len(interaction_data), dtype=np.int8)
        interaction_matrix_csr = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
        return interaction_matrix_csr

    def to_modified_interaction_matrix(self, interaction_data: pd.DataFrame, negative_interaction_value: float=-1) -> csr_matrix:
        """
        Converts a pandas.DataFrame containing user-item interactions to a scipy sparse csr matrix with matrix values based on the 'recommend' data present in the dataframe.

        If the recommend value for a user-item interaction is False, the corresponding entry in the interaction matrix is set to the negative_interaction_value parameter.
        Otherwise, the corresponding entry in the interaction matrix is set to 1.

        User and item IDs should be remapped prior to obtain a meaningful matrix.
        Duplicate interaction are dropped prior to converting to sparse matrix.

        :param interaction_data: pandas.DataFrame containing user-item interactions. Must contain 'user_id', 'item_id' and 'recommend' columns.
        :param negative_interaction_value: value assigned to interactions where 'recommend' = False
        :return: scipy.sparse.csr_matrix of the interaction data with matrix values based on the 'recommend' column.
        """


        assert 'user_id' in interaction_data.columns, "dataframe must contain 'user_id' column"
        assert 'item_id' in interaction_data.columns, "dataframe must contain 'item_id' column"
        assert 'recommend' in interaction_data.columns, "dataframe must contain 'recommend' column"

        interaction_data = interaction_data.drop_duplicates(subset=['user_id', 'item_id'])
        num_users = len(interaction_data['user_id'].unique())
        num_items = len(interaction_data['item_id'].unique())

        rows = interaction_data['user_id'].values
        cols = interaction_data['item_id'].values

        data = np.where(interaction_data['recommend'].values, 1, negative_interaction_value).astype(np.int8)
        modified_interaction_matrix_csr = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
        return modified_interaction_matrix_csr


    def to_test_fold_in_matrix(self, test_reviews: pd.DataFrame, train_item_id_mapping: dict[int, int]):
        """
        Converts a pandas.DataFrame containing fold in user-item interactions to a scipy sparse csr matrix based on the training dataset.

        Removes interactions containing items not seen in the training set.
        Remaps user IDs to consecutive integers starting from 0 so they can be used in a sparse matrix.
        Remaps item IDs to correspond to the newly assigned item IDs in the training set.

        :param test_reviews: pandas.Dataframe containing fold in user-item interactions. Must contain 'user_id' columns 'item_id'.
        :param train_item_id_mapping: dict mapping new item IDs to old item IDs for the items in the training set.
        :return: scipy.sparse.csr_matrix containing the test fold in user-item interactions.
        """

        # Filter out items not seen in the training set
        old_to_new_item_id_mapping = {v: k for k, v in train_item_id_mapping.items()}
        test_reviews = test_reviews[test_reviews['item_id'].isin(old_to_new_item_id_mapping)].copy()

        # Remap user ids
        user_ids = test_reviews['user_id'].unique()
        user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(user_ids)}
        new_to_old_test_user_id_mapping = {v: k for k, v in user_id_mapping.items()}
        test_reviews['user_id'] = test_reviews['user_id'].map(user_id_mapping)

        # update item ids to correspond to that of the training set
        test_reviews['item_id'] = test_reviews['item_id'].map(old_to_new_item_id_mapping)

        # Build CSR matrix
        test_rows = test_reviews['user_id'].values
        test_cols = test_reviews['item_id'].values
        test_data = np.ones(len(test_reviews), dtype=np.int8)

        test_fold_in_matrix_csr = csr_matrix(
            (test_data, (test_rows, test_cols)),
            shape=(len(test_reviews['user_id'].unique()), len(train_item_id_mapping))
        )

        return test_fold_in_matrix_csr, new_to_old_test_user_id_mapping