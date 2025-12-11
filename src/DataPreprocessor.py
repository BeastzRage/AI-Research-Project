import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DataPreprocessor:


    def __init__(self):
        self.cols = None
        self.rows = None
        self.num_items = None
        self.num_users = None

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

        self.num_users = len(interaction_data['old_user_id'].unique())
        self.num_items = len(interaction_data['old_item_id'].unique())

        self.rows = interaction_data['user_id'].values
        self.cols = interaction_data['item_id'].values

        return new_to_old_user_id_mapping, new_to_old_item_id_mapping


    def to_interaction_matrix(self, interaction_data: pd.DataFrame) -> csr_matrix:
        data = np.ones(len(interaction_data), dtype=np.int8)
        interaction_matrix_csr = csr_matrix((data, (self.rows, self.cols)), shape=(self.num_users, self.num_items))
        return interaction_matrix_csr

    def to_modified_interaction_matrix(self, interaction_data: pd.DataFrame, negative_interaction_value: float=-1) -> csr_matrix:
        data = np.where(interaction_data['recommend'].values, 1, negative_interaction_value).astype(np.int8)
        modified_interaction_matrix_csr = csr_matrix((data, (self.rows, self.cols)), shape=(self.num_users, self.num_items))
        return modified_interaction_matrix_csr