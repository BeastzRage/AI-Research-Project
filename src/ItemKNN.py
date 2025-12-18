import numpy as np

from tqdm import tqdm
from scipy.sparse import csr_matrix
from recpack.util import get_top_K_values

from src.metrics import calculate_ndcg
from src.StrongGeneralizationSplitter import StrongGeneralizationSplitter
from src.HelperFunctions import matrix2df, scores2recommendations

class ItemKNN:

    def __init__(self, k = None):
        assert type(k) == int or k is None, "k must be an integer"
        self.k = k


    def my_cosine_similarity(self, X: csr_matrix) -> csr_matrix:
        # returns a sparse (item x item)-matrix

        # a.b
        num = X.T @ X

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
        assert self.k is not None, "neighbor_count k never set, either on initialization or using the neighborhood_tuning method"
        cos_sim = self.my_cosine_similarity(X_train)
        cos_sim = get_top_K_values(cos_sim, self.k)

        # H . S
        scores = X_test_in @ cos_sim

        return scores.tocsr()

    def neighborhood_tuning(self, interaction_matrix_csr, train_ratio, fold_in_ratio, validation_range, step_size,
                                    split_count):

        validation_splitter = StrongGeneralizationSplitter(train_ratio=train_ratio, val_ratio=1 - train_ratio,
                                                           test_ratio=0, fold_in_ratio=fold_in_ratio)
        validation_scores: dict = {}
        for i in tqdm(range(split_count), desc="Tuning neighborhood size k"):
            train, (validation_fold_in, validation_hold_out), _, _ = validation_splitter.split(interaction_matrix_csr)
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

