import math
import random as r
import numpy as np


def simulated_annealing(energy: callable, R0: float, T: float, dt: float, s: float, ds: float, k: float, m: int, lowbound: float, seed=0.42):
    
    """
    E      : 能量函數，輸入一個解，回傳能量（越小越好）
    R0     : 初始解
    T      : 起始溫度
    dt     : 降溫係數 (0~1)
    s      : 起始擾動範圍
    ds     : 擾動範圍縮小係數 (建議 0.8~0.95)
    k      : 機率常數(類似波茲曼常數)
    m      : 每個溫度的迭代次數
    lowbound  : 停止溫度
    seed   : 亂數種子（固定後每次結果一樣，方便除錯）
    """
    
    # 初始化最佳解與最佳能量
    best_R = R0
    best_E = energy(R0)
    
    # 初始化亂數種子
    rng = r.Random(seed)

    while T >= lowbound: # 若溫度T低於停止溫度lowbound，則停止迭代
        
        for i in range(m): # 每個溫度T，迭代m次
            
            # 計算鄰近點位置(R1 會落在 [R0 − s/2, R0 + s/2])
            R1 = (R0 - s/2) + s * rng.random()
            
            # 計算能量差
            delta_E = energy(R1) - energy(R0)
            
            # 經接受新解，則將R1設為新的R0，否則回到舊解R0
            if is_accept(delta_E, T, k):
                R0 = R1
        
                # 若新解能量比最佳能量小，則更新最佳解與最佳能量
                if energy(R0) < best_E: 
                    best_R = R0
                    best_E = energy(R0)
        
            
        T *= dt # 降溫
        s *= ds # 擾動範圍縮小
            
    return best_R, best_E
  

# 計算接受新解的機率函數
def probability(delta_E: float, T: float, k: float) -> float:
    return math.exp(-delta_E / (k * T))


# 計算能量函數
def energy(x: float) -> float:
    return x**2 + 10 * math.sin(3 * x)


# 接受準則
def is_accept(delta_E: float, T: float, k: float) -> bool:
    
    bool_result = None
    
    # 決策是否接受新解
    if delta_E <= 0:            # 若新解能量比舊解能量小，則接受新解
        bool_result = True
    else:                       # 若新解能量比舊解能量大，則以機率接受新解
        p = probability(delta_E, T, k)
        bool_result = (r.random() < p) if True else False
        
    return bool_result



if __name__ == "__main__":
    
    result = simulated_annealing(
        energy, 
        R0=5, 
        T=1000, 
        dt=0.90, 
        s=10, 
        ds=0.95, 
        k=1, 
        m=100, 
        lowbound=0.001, 
        seed=42)
    
    
    best_R, best_E = result
    print(f"最佳解 x = {best_R:.4f}, E = {best_E:.4f}")