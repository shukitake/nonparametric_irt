import os
import sys

import numpy as np

from clustering.initlize_v import Opt_Init_V
from clustering.opt_clustering import Opt_clustering
from DMM.opt_emalgo import DMM_EM_Algo
from DMM.optimize_Z import Opt_Z
from MHM.heuristic_algorithm import Heu_MHM_Algo
from MHM.opt_MHM_X import Opt_MHM_X
from util.data_handling import data_handle
from util.data_visualization import data_visualization
from util.estimation_accuracy import est_accuracy
from util.log import LoggerUtil


def main(T, N, data):
    logger = LoggerUtil.get_logger(__name__)
    # パスの指定
    indpath = "/Users/shukitakeuchi/nonparametric_irt/resources"
    indpath = os.path.join(indpath, data)
    # 実験の設定
    T = T
    N = N
    # データを読み込む
    U_df, Y_df, T_true_df, icc_true_df = data_handle.pandas_read(indpath)
    # nparrayに変換
    U, init_Y, T_true, icc_true, I, J = data_handle.df_to_array(
        U_df, Y_df, T_true_df, icc_true_df
    )

    # 初期値Yを所与としてMHMについてXを解く
    """難易度行列を取得"""
    opt_MHM_X = Opt_MHM_X(U, init_Y, T)
    X_opt = opt_MHM_X.opt()

    """heu_mhm_algo = Heu_MHM_Algo(U, init_Y, T)
    X_opt, init_Y = heu_mhm_algo.repeat_process(init_Y)"""

    # 難易度行列推定
    logger.info("estimation Z")
    opt_Z = Opt_Z(U, T)
    init_Z = opt_Z.Est_Diff_Rank(X_opt)

    # 初期クラスタ行列
    """クラスター数に応じて上から分割"""
    opt_init_V = Opt_Init_V(U, N, T)
    init_V = opt_init_V.initialize_V(init_Z)

    # クラスタリング
    logger.info("clustering")
    """初期推定値VとしてMHMの順序行列を用いる"""
    opt_cl = Opt_clustering(U, init_Y, init_V, N, T)
    V_opt = opt_cl.opt()
    # data_visualization.cl_icc_show(W_opt, V_opt, J, N, T)

    # DMM
    logger.info("DMM(W,Y) start")
    opt_DMM = DMM_EM_Algo(U, init_Y, init_Z, V_opt, N, T)
    W_opt, Y_opt = opt_DMM.repeat_process()
    T_est = est_accuracy.show_class(Y_opt)
    # 精度
    rmse_class = est_accuracy.rmse_class(T_true, T_est)
    logger.info(f"rmse_class:{rmse_class}")
    rmse_icc = est_accuracy.rmse_icc(icc_true, init_Z, W_opt)
    logger.info(f"rmse_icc:{rmse_icc}")
    # data_visualization.cluster_icc(W_opt, init_Z, V_opt, J, N, T)
    return W_opt, Y_opt, init_Z, rmse_class, rmse_icc


if __name__ == "__main__":
    for num_item in [60]:
        for num_user in [300]:
            for ratio in [20, 50]:
                for seed in range(10):
                    result = np.empty((0, 3))
                    outdpath = "/workspaces/irt_pro/resources"
                    for n in [num_item, 10, 5, 3, 1]:
                        data = os.path.join(
                            f"data{seed}", f"{num_item}*{num_user}*{ratio}"
                        )
                        T = 10
                        N = n
                        W, Y, Z, rmse_class, rmse_icc = main(T, N, data)
                        # data_visualization.DMM_icc_show(W, Z, J, T)
                        result = np.vstack([result, [n, rmse_class, rmse_icc]])
                    outdpath = os.path.join(outdpath, data, "output.csv")
                    np.savetxt(outdpath, result, delimiter=",", fmt="%.5f")
