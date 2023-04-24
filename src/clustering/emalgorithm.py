import sys
import numpy as np
from clustering.optimize_w import Opt_W
from util.log import LoggerUtil


class EM_Algo:
    def __init__(self, U, Y, V, N, T):
        self.U = U
        """Yは初期推定値を使う"""
        self.init_Y = Y
        """Vは初期推定値、MHMの難易度でクラスタリング"""
        self.init_V = V
        self.I, self.J = np.shape(self.U)
        self.T = T
        self.N = N
        self.logger = LoggerUtil.get_logger(__name__)
        return

    @classmethod
    def con_prob(cls, W_nt, Y_it, U_ij):
        return np.power(
            np.power(W_nt, U_ij) * np.power(1 - W_nt, 1 - U_ij), Y_it
        )

    def convert_V_cluster(self, V):
        index = np.argmax(V, axis=1)
        V = np.zeros((self.J, self.N))
        for i in range(len(index)):
            V[i, index[i]] = 1
        return V

    def EStep(self, pi, W):
        # Vの更新
        f = np.array(
            [
                [
                    np.prod(
                        [
                            EM_Algo.con_prob(
                                W[n, t], self.init_Y[i, t], self.U[i, j]
                            )
                            for i in range(self.I)
                            for t in range(self.T)
                        ]
                    )
                    for n in range(self.N)
                ]
                for j in range(self.J)
            ]
        )
        f1 = np.multiply(pi, f)
        f2 = np.sum(f1, 1).reshape(-1, 1)
        V = np.divide(f1, f2)
        V_opt = EM_Algo.convert_V_cluster(self, V)
        return V, V_opt

    def MStep(self, V):
        # piの更新
        pi = np.sum(V, axis=0) / self.J

        # Wの更新
        opt_W = Opt_W(self.U, self.init_Y, V, self.N, self.T)
        opt_W.modeling()
        W_opt, obj = opt_W.solve()
        W_opt = np.reshape(W_opt, [self.N, self.T])
        return pi, W_opt

    def repeat_process(self):
        # 初期ステップ -> MStep
        i = 1
        self.logger.info("first step")
        # Vを初期化
        V_opt = self.init_V
        pi, W = EM_Algo.MStep(self, V_opt)
        est_V = np.zeros((self.J, self.N))
        while np.any(est_V != V_opt):
            est_V = V_opt
            # 繰り返し回数
            i += 1
            self.logger.info(f"{i}th step")
            # EStep
            V, V_opt = EM_Algo.EStep(self, pi, W)
            # MStep
            pi, W = EM_Algo.MStep(self, V)
            # 収束しない時、30回で終了させる
            if i == 30:
                return W, V_opt
        self.logger.info("finish")
        return W, V_opt
