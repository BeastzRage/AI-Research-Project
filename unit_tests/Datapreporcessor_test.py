import pandas as pd
from src.DataPreprocessor import DataPreprocessor


data = {'user_id': [3, 3, 4, 7 ,9], 'item_id': [0, 0, 1, 8, 500], 'recommend': [True, True, False, True, False]}
interaction_data = pd.DataFrame.from_dict(data)

data_preprocessor = DataPreprocessor()
new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.map_ids(interaction_data)


assert new_to_old_user_id_mapping == {0:3, 1:4, 2:7, 3:9}
assert new_to_old_item_id_mapping == {0:0, 1:1, 2:8, 3:500}
assert list(interaction_data['user_id']) == [0, 0, 1, 2, 3]
assert list(interaction_data['item_id']) == [0, 0, 1, 2, 3]
assert list(interaction_data['old_user_id']) == [3, 3, 4, 7, 9]
assert list(interaction_data['old_item_id']) == [0, 0, 1, 8, 500]

interaction_matrix = data_preprocessor.to_interaction_matrix(interaction_data)
matrix_data = interaction_matrix.tocoo().toarray().tolist()
assert matrix_data == [[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]]

modified_interaction_matrix = data_preprocessor.to_modified_interaction_matrix(interaction_data, -1)
modified_matrix_data = modified_interaction_matrix.tocoo().toarray().tolist()
assert modified_matrix_data == [[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, -1]]

test_in_data = {'user_id': [11, 17, 17, 80, 88], 'item_id': [500, 0, 77, 1, 77]}
test_in_interaction_data = pd.DataFrame.from_dict(test_in_data)
test_in_matrix, test_new_to_old_user_id_mapping = data_preprocessor.to_test_fold_in_matrix(test_in_interaction_data, new_to_old_item_id_mapping)

test_data = test_in_matrix.tocoo().toarray().tolist()
assert test_data == [[0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0]]


data = {'users': [3, 4, 7 ,9], 'item_id': [0, 1, 8, 500]}
wrong_user_column_name_data = pd.DataFrame.from_dict(data)

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.map_ids(wrong_user_column_name_data)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'user_id' column"
except Exception as e:
    raise e

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_interaction_matrix(wrong_user_column_name_data)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'user_id' column"
except Exception as e:
    raise e

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_modified_interaction_matrix(wrong_user_column_name_data)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'user_id' column"
except Exception as e:
    raise e

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_test_fold_in_matrix(wrong_user_column_name_data,{})
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'user_id' column"
except Exception as e:
    raise e



data = {'user_id': [3, 4, 7 ,9], 'items_id': [0, 1, 8, 500]}
wrong_user_column_name_data = pd.DataFrame.from_dict(data)

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.map_ids(wrong_user_column_name_data)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'item_id' column"
except Exception as e:
    raise e

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_interaction_matrix(wrong_user_column_name_data)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'item_id' column"
except Exception as e:
    raise e

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_modified_interaction_matrix(wrong_user_column_name_data)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'item_id' column"
except Exception as e:
    raise e

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_test_fold_in_matrix(wrong_user_column_name_data,{})
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'item_id' column"
except Exception as e:
    raise e



data = {'user_id': [3, 4, 7 ,9], 'item_id': [0, 1, 8, 500]}
interaction_data = pd.DataFrame.from_dict(data)

try:
    new_to_old_user_id_mapping, new_to_old_item_id_mapping = data_preprocessor.to_modified_interaction_matrix(interaction_data,-1)
    raise AssertionError("Should have raised AssertionError for absent column")
except AssertionError as e:
    assert str(e) == "dataframe must contain 'recommend' column"
except Exception as e:
    raise e