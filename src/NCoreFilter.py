import numpy as np
from scipy.sparse import csr_matrix

class NCoreFilter:

    def __init__(self, N = 5):
        # assert N > 0, "N must be greater than 0."
        self.N = N


    def filter(self, interaction_matrix_csr: csr_matrix, item_mapping, user_mapping, ):
        filtered_interaction_matrix, updated_user_mapping = self.MinItemsPerUser(interaction_matrix_csr, user_mapping)
        filtered_interaction_matrix, updated_item_mapping = self.MinUsersPerItem(filtered_interaction_matrix, item_mapping)
        return filtered_interaction_matrix, updated_item_mapping, updated_user_mapping

    def MinUsersPerItem(self, interaction_matrix_csr: csr_matrix, item_mapping):
        # Filter out items with strictly less than N interactions
        item_mask = interaction_matrix_csr.getnnz(axis=0) >= self.N

        # Apply the mask to the matrix to filter items
        filtered_interaction_matrix = interaction_matrix_csr[:, item_mask]

        # Update the mapping
        item_ids = np.array(list(item_mapping.values()))
        good_item_ids = item_ids[item_mask]
        updated_item_mapping = {new_id: old_id for new_id, old_id in enumerate(good_item_ids)}

        return filtered_interaction_matrix, updated_item_mapping

    def MinItemsPerUser(self, interaction_matrix_csr:csr_matrix, user_mapping):
        # Filter out users with strictly less than N interactions
        user_mask = interaction_matrix_csr.getnnz(axis=1) >= self.N

        # Apply the mask to the matrix to filter users
        filtered_interaction_matrix = interaction_matrix_csr[user_mask]

        # Update the mapping
        user_ids = np.array(list(user_mapping.values()))
        good_user_ids = user_ids[user_mask]
        updated_user_mapping = {new_id: old_id for new_id, old_id in enumerate(good_user_ids)}

        return filtered_interaction_matrix, updated_user_mapping
