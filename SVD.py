import pandas as pd
import numpy as np
from numpy import linalg

class SVD:

    def __init__(self, dataframe: pd.DataFrame):

        self.dataframe : pd.DataFrame = dataframe
        self.U ,self.Sigma, self.V_t = linalg.svd(self.dataframe)
        self.U_reduced, self.Vt_reduced, self.Sigma_reduced = self.reduce_rank()
        self.wawa = pd.DataFrame(self.U_reduced * self.Vt_reduced * self.Sigma_reduced)

    def reduce_rank(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        total_sum = np.sum(self.Sigma)
        sum = 0
        """
        values in Sigma are the weights representing amount of info.
        after reduction, remaining weights should add up to at least 90% per Falk, Chapter 11 Finding hidden genres with matrix factorization
        """
        for k in range(self.Sigma.shape[0]):
            sum += self.Sigma[k]
            if sum >= total_sum*1:
                break

        U_reduced = np.mat(self.U[:, :k])
        Vt_reduced = np.mat(self.V_t[:k, :])
        Sigma_reduced = Sigma_reduced = np.eye(k) * self.Sigma[:k]
        return U_reduced, Sigma_reduced, Vt_reduced,


    def use_baseline(self):
        global_mean = self.dataframe[self.dataframe > 0].mean().mean()
        M_minus_mean = self.dataframe[self.dataframe > 0] - global_mean
        user_bias = M_minus_mean.T.mean()
        item_bias = M_minus_mean.apply(lambda r: r - user_bias).mean()

        df1 = pd.DataFrame({"A": user_bias.tolist()})
        df2 = pd.DataFrame({"B": item_bias.to_list()})

        baseline_values = df1["A"].values[:, None] + df2["B"].values[None, :] + global_mean
        baseline_dataframe = pd.DataFrame(baseline_values, index=df1.index, columns=df2.index)

        df_result = self.dataframe.copy()
        mask = df_result.to_numpy() == 0
        df_result = df_result.astype(float)
        df_result.iloc[:, :] = np.where(mask, baseline_values, df_result.to_numpy())

        self.dataframe = df_result

    def get_scores(self, test_fold_in: pd.DataFrame) -> pd.DataFrame:
        V = self.V_t.transpose()
        U_test = (test_fold_in.to_numpy() @ V) / self.Sigma



        return pd.DataFrame((U_test  * self.Sigma) @ self.V_t)

