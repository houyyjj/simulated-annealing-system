import math
import random as r
import numpy as np
import energy_and_solution_ic as ic




def simulated_annealing_steps(
    energy, T, dt, s, ds, k, m, lowbound,
    seed=0.42, *, initial=None,
):
    """逐輪提供目前解與最佳解；每列保留 [x, y, w, h]（含旋轉後寬高）。"""
    # 網頁有提供初始解就沿用；使用浮點副本，避免修改外部佈局或截斷移動量。
    R0 = np.array(ic.initial_solution() if initial is None else initial, dtype=float, copy=True)
    if R0.shape != (len(ic.MODULES), 4):
        raise ValueError("佈局每列須包含 [x, y, w, h]，並與 IC 模組數量一致。")

    current_E = best_E = energy(R0)
    best_R = R0.copy()
    rng = r.Random(seed)

    # 第一次提供初始狀態，尚未開始搜尋。
    yield R0.copy(), current_E, best_R.copy(), best_E, T, s

    while T >= lowbound:
        for _ in range(m):
            # 直接沿用 IC 的移動、交換與 90 度旋轉，不以原始尺寸覆蓋寬高。
            R1 = ic.neighbor(R0, s)
            candidate_E = energy(R1)
            delta_E = candidate_E - current_E

            if is_accept(delta_E, T, k, rng=rng):
                R0 = R1
                current_E = candidate_E

                # 統一使用傳入的成本函式，保留網頁設定的權重。
                if current_E < best_E:
                    best_R = R0.copy()
                    best_E = current_E

        T *= dt
        s *= ds

        # 每完成一輪，就暫停並提供結果。
        yield R0.copy(), current_E, best_R.copy(), best_E, T, s


def simulated_annealing(
    energy, T, dt, s, ds, k, m, lowbound,
    seed=0.42, *, initial=None,
):
    # 一般程式需要一次算完時，把所有輪次執行完即可。
    for layout, current_E, best_R, best_E, temperature, radius in (
        simulated_annealing_steps(
            energy, T, dt, s, ds, k, m, lowbound,
            seed=seed, initial=initial,
        )
    ):
        pass

    return best_R, best_E


# 計算接受新解的機率函數
def probability(delta_E: float, T: float, k: float) -> float:
    return math.exp(-delta_E / (k * T))

# 接受準則
def is_accept(delta_E: float, T: float, k: float, *, rng=r) -> bool:
    return delta_E <= 0 or rng.random() < probability(delta_E, T, k)
