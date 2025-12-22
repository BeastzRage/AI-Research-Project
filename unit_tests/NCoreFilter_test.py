from src.NCoreFilter import NCoreFilter
from scipy.sparse import csr_matrix

data = [[0,0,1,0,0],
        [1,0,1,1,1],
        [1,0,1,0,1],
        [0,1,1,0,1],
        [1,0,0,0,0]]

two_core_filer = NCoreFilter(2)

filtered_matrix, updated_item_mapping, updated_user_mapping = two_core_filer.filter(csr_matrix(data), {0:0, 1:1, 2:2, 3:3, 4:4}, {0:0, 1:1, 2:2, 3:3, 4:4})

filtered_data = filtered_matrix.toarray().tolist()

assert filtered_data == [[1,1,1],
                         [1,1,1],
                         [0,1,1]]
assert updated_item_mapping == {0:0, 1:2, 2:4}
assert updated_user_mapping == {0:1, 1:2, 2:3}


try:
    zero_core_filter = NCoreFilter(0)
    raise AssertionError("Should have raised an exception")
except AssertionError as e:
    assert str(e) == "N must be greater than 0."
except Exception as e:
    raise e