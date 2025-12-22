from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter
from scipy.sparse import csr_matrix

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

data = [[0,0,1,0,0],
        [1,0,1,1,1],
        [1,0,1,0,1],
        [0,1,1,0,1],
        [1,0,0,0,0]]

strong_gen_splitter = StrongGeneralizationSplitter(0.6, 0.2 ,0.2, 0.8)


train_matrix, (val_fold_in, val_hold_out), (test_fold_in, test_hold_out), (train_users, val_users, test_users) = strong_gen_splitter.split(csr_matrix(data))


assert set(train_users).isdisjoint(val_users), "Train and validation sets overlap"
assert set(train_users).isdisjoint(test_users), "Train and test sets overlap"
assert set(val_users).isdisjoint(test_users), "Validation and test sets overlap"


assert check_interaction_overlap(val_fold_in, val_hold_out), "Overlap in validation fold in and hold out set"
assert check_interaction_overlap(test_fold_in, test_hold_out), "Overlap in test fold in and hold out set"


try:
    strong_gen_splitter = StrongGeneralizationSplitter(train_ratio=1.1)
except AssertionError as e:
    assert str(e) == "Train, validation, and test ratios must sum to 1."
except Exception as e:
    raise e

try:
    strong_gen_splitter = StrongGeneralizationSplitter(val_ratio=1.1)
except AssertionError as e:
    assert str(e) == "Train, validation, and test ratios must sum to 1."
except Exception as e:
    raise e

try:
    strong_gen_splitter = StrongGeneralizationSplitter(test_ratio=1.1)
except AssertionError as e:
    assert str(e) == "Train, validation, and test ratios must sum to 1."
except Exception as e:
    raise e

try:
    strong_gen_splitter = StrongGeneralizationSplitter(0.4, 0.5, 0.11)
except AssertionError as e:
    assert str(e) == "Train, validation, and test ratios must sum to 1."
except Exception as e:
    raise e



try:
    strong_gen_splitter = StrongGeneralizationSplitter(fold_in_ratio=0)
except AssertionError as e:
    assert str(e) == "Fold in ratio must be between 0 and 1."
except Exception as e:
    raise e


try:
    strong_gen_splitter = StrongGeneralizationSplitter(fold_in_ratio=1)
except AssertionError as e:
    assert str(e) == "Fold in ratio must be between 0 and 1."
except Exception as e:
    raise e
