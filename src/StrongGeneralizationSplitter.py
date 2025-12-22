import numpy as np
from scipy.sparse import lil_matrix


class StrongGeneralizationSplitter:
    def __init__(self, train_ratio=0.8, val_ratio=0, test_ratio=0.2, fold_in_ratio=0.8):
        """
        Strong Generalization Splitter constructor

        :param train_ratio: ratio of interaction data that should make up the training set
        :param val_ratio: ratio of interaction data that should make up the validation set
        :param test_ratio: ratio of interaction data that should make up the test set
        :param fold_in_ratio: ratio of test/validation data that should make up the fold in test/validation set
        """
        assert train_ratio + val_ratio + test_ratio == 1, "Train, validation, and test ratios must sum to 1."
        assert 0 < fold_in_ratio < 1, "Fold in ratio must be between 0 and 1."
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.fold_in_ratio = fold_in_ratio
        self.hold_out_ratio = 1 - fold_in_ratio

    def split(self, interaction_matrix_csr):
        """
        Makes a strong generalization split of the sparce interaction matrix
        :param interaction_matrix_csr: scipy.sparse.csr_matrix interaction matrix to be split
        :return: tuple(train_matrix, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out)
            train_matrix scipy.sparse.csr_matrix training data matrix
            val_fold_in scipy.sparse.csr_matrix validation fold in data matrix
            val_hold_out scipy.sparse.csr_matrix validation hold out data matrix
            test_fold_in scipy.sparse.csr_matrix test fold in data matrix
            test_hold_out scipy.sparse.csr_matrix test hold out data matrix
            train_users list of users indexes in the train split
            val_users list of users indexes in the validation split
            test_users list of users indexes in the test split
        """

        num_users, num_items = interaction_matrix_csr.shape

        # Shuffle users for random splitting
        users = np.arange(num_users)
        np.random.shuffle(users)

        # Split users into training, validation, and test sets based on the specified ratios
        train_end = int(num_users * self.train_ratio)
        val_end = train_end + int(num_users * self.val_ratio)

        train_users = users[:train_end]
        val_users = users[train_end:val_end]
        test_users = users[val_end:]

        train_matrix = interaction_matrix_csr[train_users, :]
        val_matrix = interaction_matrix_csr[val_users, :]
        test_matrix = interaction_matrix_csr[test_users, :]

        # Create fold-in and hold-out splits for validation and test sets
        val_fold_in, val_hold_out = self.split_interactions(val_matrix)
        test_fold_in, test_hold_out = self.split_interactions(test_matrix)

        return train_matrix, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users)

    def split_interactions(self, interaction_matrix_csr):
        """
        Splits interactions matrix into a fold in and hold out matrices
        :param interaction_matrix_csr: scipy.sparse.csr_matrix interaction matrix to be split
        :return: tuple(fold_in_matrix, hold_out_matrix)
            fold_in_matrix scipy.sparse.csr_matrix fold in matrix
            hold_out_matrix scipy.sparse.csr_matrix hold out matrix
        """


        # Convert the matrix to lil format for easier manipulation
        lil = interaction_matrix_csr.tolil()

        # Initialize the lists to store fold-in and hold-out data
        fold_in_lil = lil_matrix((interaction_matrix_csr.shape[0], interaction_matrix_csr.shape[1]))
        hold_out_lil = lil_matrix((interaction_matrix_csr.shape[0], interaction_matrix_csr.shape[1]))

        # Iterate over each row (user)
        for i in range(lil.shape[0]):
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