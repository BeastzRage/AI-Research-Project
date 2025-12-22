import numpy as np
from scipy.sparse import csr_matrix

class NCoreFilter:

    def __init__(self, N = 5):
        """
        constructor for NCoreFilter
        :param N: minimum number of interactions a user/item will have after applying the filter, must be a positive integer
        """

        assert N > 0, "N must be greater than 0."
        self.N = N


    def filter(self, interaction_matrix_csr: csr_matrix, item_mapping, user_mapping, ):
        """
        filters a sparse user-item matrix such that each user has interacted with at least N items and vice versa.

        the user and item mappings are updated accordingly.

        :param interaction_matrix_csr: scipy.sparse.csr_matrix sparce user-item interaction matrix
        :param item_mapping: dict the interaction matrix's new to original item ID mapping
        :param user_mapping: dict the interaction matrix's new to original user ID mapping
        :return: tuple(filtered_interaction_matrix, updated_item_mapping, updated_user_mapping)
            scipy.sparse.csr_matrix filtered_interaction_matrix is the filtered sparce interaction matrix
            dict updated_item_mapping is the updated new to old item ID mapping
            dict updated_user_mapping is the updated new to old user ID mapping
        """

        filtered_interaction_matrix, updated_user_mapping = self.MinItemsPerUser(interaction_matrix_csr, user_mapping)
        filtered_interaction_matrix, updated_item_mapping = self.MinUsersPerItem(filtered_interaction_matrix, item_mapping)
        return filtered_interaction_matrix, updated_item_mapping, updated_user_mapping

    def MinUsersPerItem(self, interaction_matrix_csr: csr_matrix, item_mapping):
        """
        Filters a sparse user-item interaction matrix such that each item has been interacted with by at least N users.

        Items with strictly fewer than N non-zero interactions are removed.
        The item ID mapping is updated accordingly.

        :param interaction_matrix_csr: scipy.sparse.csr_matrix  Sparse user-item interaction matrix
        :param item_mapping: dict mapping from new item IDs to original item IDs
        :return: tuple(filtered_interaction_matrix, updated_item_mapping)
            scipy.sparse.csr_matrix filtered_interaction_matrix is the filtered sparse interaction matrix with low-interaction items removed
            dict updated_item_mapping is the updated new to old item ID mapping
        """

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
        """
        filters a sparse user-item interaction matrix such that each user has
        interacted with at least N items.

        users with strictly fewer than N non-zero interactions are removed.
        the user ID mapping is updated accordingly.

        :param interaction_matrix_csr: scipy.sparse.csr_matrix Sparse user-item interaction matrix
        :param user_mapping: dict Mapping from new user IDs to original user IDs
        :return: tuple(filtered_interaction_matrix, updated_user_mapping)
            scipy.sparse.csr_matrix filtered_interaction_matrix is the filtered sparse interaction matrix with low-interaction users removed
            dict updated_user_mapping is the updated new to old user ID mapping
        """

        # Filter out users with strictly less than N interactions
        user_mask = interaction_matrix_csr.getnnz(axis=1) >= self.N

        # Apply the mask to the matrix to filter users
        filtered_interaction_matrix = interaction_matrix_csr[user_mask]

        # Update the mapping
        user_ids = np.array(list(user_mapping.values()))
        good_user_ids = user_ids[user_mask]
        updated_user_mapping = {new_id: old_id for new_id, old_id in enumerate(good_user_ids)}

        return filtered_interaction_matrix, updated_user_mapping
