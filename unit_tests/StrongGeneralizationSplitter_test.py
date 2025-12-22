from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter
from scipy.sparse import csr_matrix

data = [[0,0,1,0,0],
        [1,0,1,1,1],
        [1,0,1,0,1],
        [0,1,1,0,1],
        [1,0,0,0,0]]

strong_gen_splitter = StrongGeneralizationSplitter(0.6, 0.2 ,0.2, 0.8)

train_matrix, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = strong_gen_splitter.split(csr_matrix(data))


def check_interaction_overlap(fold_in_matrix, hold_out_matrix):
    """
    Checks for overlapping interactions in the fold in and hold out data.
    :param fold_in_matrix: fold in data matrix
    :param hold_out_matrix: hold out data matrix
    :return: True if no overlap, False otherwise
    """
    for user in range(fold_in_matrix.shape[0]):
        if set(hold_out_matrix[user].indices).intersection(set(fold_in_matrix[user].indices)):
            return False
    return True


assert set(train_users).isdisjoint(val_users), "Train and validation sets overlap"
assert set(train_users).isdisjoint(test_users), "Train and test sets overlap"
assert set(val_users).isdisjoint(test_users), "Validation and test sets overlap"


assert check_interaction_overlap(val_fold_in, val_hold_out), "Overlap in validation fold in and hold out set"
assert check_interaction_overlap(test_fold_in, test_hold_out), "Overlap in test fold in and hold out set"