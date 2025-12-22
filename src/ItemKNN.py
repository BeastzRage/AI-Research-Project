import numpy as np

from tqdm import tqdm
from scipy.sparse import csr_matrix
from recpack.util import get_top_K_values

from src.metrics import calculate_ndcg
from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter
from src.HelperFunctions import matrix2df, scores2recommendations

class ItemKNN:

    def __init__(self, k = None):
        """
        initialization for ItemKNN
        :param k: neighborhood size, must be an integer greater than 0
        """
        assert (type(k) == int and k > 0 ) or k is None, "k must be an integer greater than 0"
        self.k = k


    def my_cosine_similarity(self, X: csr_matrix) -> csr_matrix:
        """
        Computes the cosine similarity between each pair of items found in a user-item interaction matrix.

        :param X: scipy.sparse.csr_matrix sparse user-item interaction matrix
        :return: scipy.sparse.csr_matrix cosine similarity item x item matrix
        """

        # ||a||2.||b||2
        norms = np.sqrt(np.array(X.power(2).sum(axis=0))).ravel()

        # dont devide by 0
        norms[norms == 0] = 1e-10

        # cos(a,b)
        X_normed = X.multiply(1 / norms)
        cos_sim = X_normed.T @ X_normed

        # set diagonal to 0
        cos_sim.setdiag(0)
        cos_sim.eliminate_zeros()

        return cos_sim.tocsr()

    def item_knn_scores(self, X_train: csr_matrix, X_test_in: csr_matrix) -> csr_matrix:
        """
        Computes the itemknn score of each item for each user in a fold in test interaction matrix.

        The training matrix is used to compute the cosine similarity matrix which is then used to score each item for the users.
        If the neighbor_count k was not set on class initialization, neighborhood_tuning must be run prior to calling this function.


        :param X_train: scipy.sparse.csr_matrix sparse user-item training interaction matrix to compute similarity scores
        :param X_test_in: scipy.sparse.csr_matrix sparse user-item test fold in interaction matrix for which to compute itemknn scores
        :return: scipy.sparse.csr_matrix containing itemknn scores of each item for each user
        """


        assert self.k is not None, "neighbor_count k never set, either on initialization or using the neighborhood_tuning method"
        cos_sim = self.my_cosine_similarity(X_train)
        cos_sim = get_top_K_values(cos_sim, self.k)

        # H . S
        scores = X_test_in @ cos_sim

        return scores.tocsr()

    def neighborhood_tuning(self, interaction_matrix_csr, train_ratio, fold_in_ratio, validation_range, step_size,
                                    split_count):
        """
        Tests multiple different values for the neighborhood size k on the interaction data and sets the algorithms neighborhood size to value with
        the highest average NDCG@20 score.

        The data will be split into training and validation sets. This is done 'split_count' number of times. For each random split, the NDCG@20 score is computed for each k.
        In the end, the average NDCG@20 score is computed for each k and the highest scoring value for k is picked.

        :param interaction_matrix_csr: scipy.sparse.csr_matrix interaction matrix to compute NDCG@20 scores with
        :param train_ratio: ratio of data that is used as training data, 1 - train_ratio is the validation ratio
        :param fold_in_ratio: the ratio of data in the validation set to be used as fold in test data, 1 - fold_in_ratio is the hold out ratio
        :param validation_range: amount of different neighborhood size values to test
        :param step_size: difference between neighborhood size values to test
        :param split_count: amount of random splits that should be tested
        :return: int the highest average scoring neighborhood size k
        """


        validation_splitter = StrongGeneralizationSplitter(train_ratio=train_ratio, val_ratio=1 - train_ratio,
                                                           test_ratio=0, fold_in_ratio=fold_in_ratio)
        validation_scores: dict = {}
        for i in tqdm(range(split_count), desc="Tuning neighborhood size k"):
            train, (validation_fold_in, validation_hold_out), _ = validation_splitter.split(interaction_matrix_csr)
            k = 1
            for j in range(validation_range):
                # dataframe version of hold-out set to compute metrics
                df_validation_out = matrix2df(validation_hold_out)

                algo = ItemKNN(k)
                scores = algo.item_knn_scores(train, validation_fold_in)
                df_recos = scores2recommendations(scores, validation_fold_in, 20)

                ndcg = calculate_ndcg(df_recos, 20, df_validation_out)
                if k in validation_scores:
                    validation_scores[k] += ndcg
                else:
                    validation_scores[k] = ndcg
                k += step_size

        average_validation_scores = {k: v / split_count for k, v in validation_scores.items()}
        best_average_k = max(average_validation_scores, key=average_validation_scores.get)
        print("Best k: ", best_average_k)
        self.k = best_average_k
        return best_average_k

