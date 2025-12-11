import numpy as np
from scipy.sparse import csr_matrix
from recpack.util import get_top_K_values

class ItemKNN:

    @staticmethod
    def my_cosine_similarity(X: csr_matrix) -> csr_matrix:
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

    @staticmethod
    def item_knn_scores(
            X_train: csr_matrix,
            X_test_in: csr_matrix,
            neighbor_count: int
    ) -> csr_matrix:
        cos_sim = ItemKNN.my_cosine_similarity(X_train)
        cos_sim = get_top_K_values(cos_sim, neighbor_count)

        # H . S
        scores = X_test_in @ cos_sim

        return scores.tocsr()