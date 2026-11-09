import numpy as np 

# 1.基本參數設定
#晶片最大邊界長寬(100*100)
BOUND_X = 100.0
BOUND_Y = 100.0

#成本函數三個權重: Cost = ALPHA*Area + BETA*Σ(Wij*HPWLij) + GAMMA*Overlap
ALPHA = 1.0   #面積權重 α
BETA = 1.0    #線長延遲權重 β
OVERLAP_PENALTY = 10.0  #重疊懲罰權重 γ (重疊1罰10)

#IC模組清單[w寬,h高] -> 4*2 (0:CPU, 1:GPU, 2:I/O, 3:Cache)
MODULES = np.array(
    [
        [10.0,10.0],
        [10.0,10.0],
        [10.0,10.0],
        [10.0,10.0]
        ]
    )

#模組連線頻率權重 Wij
#對應下方 triu_indices 配對順序: (0,1)CPU-GPU, (0,2)CPU-I/O, (0,3)CPU-Cache, (1,2)GPU-I/O, (1,3)GPU-Cache, (2,3)I/O-Cache
W_IJ = np.array([0.0, 2.0, 10.0, 1.0, 5.0, 0.0])

# 2.產生初始解R0(step1)
def initial_solution():
    #4顆方塊的x,y最大合法邊界
    max_x = BOUND_X - MODULES[:,0]
    max_y = BOUND_Y - MODULES[:,1]

    #隨機在合法範圍內抽出4顆方塊的x,y座標
    x = np.random.uniform(0.0,max_x)
    y = np.random.uniform(0.0,max_y)
    
    #合併直排x,直排y
    R0 = np.column_stack((x,y,MODULES[:, 0], MODULES[:, 1]))
    #numpy 二維(4,4)，每列是一顆模組的[x,y,w,h]
    return R0

# 3: 由現解 R 產生鄰近解R1(step3:隨機移動一顆方塊)
def neighbor(R,s):
    #隨機抽0,1,2 ->選擇哪種擾動
    mode = np.random.randint(0,3)
    
    #mode = 0(移動)
    R_move = R.copy()
    # 從0,1,2,3中隨機抽一個號碼idx決定這次要移動哪一顆模組
    idx_m = np.random.randint(0,len(R))
    #取被抽取的模組w,h
    w,h = R_move[idx_m,2],R_move[idx_m,3]
    
    #各自產生0.0~1.0之間的隨機亂數 rand(1)
    rand_x = np.random.rand()
    rand_y = np.random.rand()
    
    #新座標 = 舊座標 - (擾動範圍s/2)+s*亂數
    #R1[idx,0] 第idx顆的x座標
    new_x = R_move[idx_m,0] - (s/2.0) + s*rand_x
    new_y = R_move[idx_m,1] - (s/2.0) + s*rand_y
    
    #np.clip(數值,最小值,最大值)->限制範圍
    R_move[idx_m,0] = np.clip(new_x,0.0,BOUND_X - w)
    R_move[idx_m,1] = np.clip(new_y,0.0,BOUND_Y - h)

    #mode = 1(交換)
    R_swap = R.copy()
    idx1,idx2 = np.random.choice(len(R),2,replace=False)
    #這兩顆的 x,y 座標(第0,1欄，0:2)對調；第2,3欄的 w,h 與列順序不變
    R_swap[[idx1,idx2],0:2] = R_swap[[idx2,idx1],0:2]
    R_swap[:,0] = np.clip(R_swap[:,0],0.0,BOUND_X-R_swap[:,2])
    R_swap[:,1] = np.clip(R_swap[:,1],0.0,BOUND_Y-R_swap[:,3])
    
    #mode = 2(旋轉90度)
    R_rotate = R.copy()
    idx_r = np.random.randint(0,len(R))
    # 旋轉90度 = 模組的 寬(第2欄) 跟 高(第3欄) 數值互換
    R_rotate[idx_r,[2,3]] = R_rotate[idx_r,[3,2]]
    # 轉後長寬改變(原本貼在右邊界，轉完變寬會凸出去)->重 clip 推回合法邊界內
    R_rotate[idx_r,0] = np.clip(R_rotate[idx_r,0],0.0,BOUND_X-R_rotate[idx_r,2])    
    R_rotate[idx_r,1] = np.clip(R_rotate[idx_r,1],0.0,BOUND_Y-R_rotate[idx_r,3])
    
    candidates = np.array([R_move,R_swap,R_rotate])
    return candidates[mode]

# 4.count 能量函數 E(R) (step1&4:算排的好壞)
# 能量越低->排越好：Cost = α*Area + β*Σ(Wij*HPWLij) + γ*Overlap
def energy(R):
    # R[:, 0]->左邊界 x；加上 R[:, 2](當下的寬度 w)=右邊界
    min_x = np.min(R[:,0])
    max_x = np.max(R[:,0]+R[:,2])
    # R[:, 1]->下邊界 y；加上 R[:, 3](當下的高度 h)=上邊界
    min_y = np.min(R[:,1])
    max_y = np.max(R[:,1]+R[:,3])
    
    # 1.大外框面積 Area = 總寬度(最右 - 最左) * 總高度(最上 - 最下)
    box_area = (max_x - min_x) * (max_y - min_y)
    
    #自動配對
    #n=4為例->i:[0,0,0,1,1,2] j:[1,2,3,2,3,3]
    n = len(R)
    # n：矩陣的列數（或方陣大小）。
    # k（可選）：對角線偏移量（預設 `0` 為主對角線；`> 0` 往上偏移；`< 0` 往下偏移）。
    # m（可選）：矩陣的欄數（預設等於 `n`，即適用於 n × n 方陣）。
    i,j = np.triu_indices(n,k=1)
    
    #根據配對號碼i抓左邊方塊的x1,y1 與當下的寬高 w1,h1(第2,3欄)
    x1,y1 = R[i,0],R[i,1]
    w1,h1 = R[i,2],R[i,3]
    #j，抓右邊方塊
    x2,y2 = R[j,0],R[j,1]
    w2,h2 = R[j,2],R[j,3]
    
    # 2.count HPWL線長：兩顆模組中心點的 寬差(|cx1-cx2|) + 高差(|cy1-cy2|)
    cx1,cy1 = x1 + (w1/2.0), y1 + (h1/2.0)
    cx2,cy2 = x2 + (w2/2.0), y2 + (h2/2.0)
    hpwl = np.abs(cx1 - cx2) + np.abs(cy1 - cy2)
    # 乘上通訊頻率權重 Wij 後全部加總
    wire_cost = np.sum(W_IJ * hpwl)
    
    # 3.count重疊面積：小右界np.minimum(右1,右2)-大左界np.maximum(左1,左2)
    overlap_w = np.minimum(x1+w1,x2+w2)-np.maximum(x1,x2)
    overlap_h = np.minimum(y1+h1,y2+h2)-np.maximum(y1,y2)
    
    # 如果沒重疊，上面減出來會是負數；用 np.maximum 跟 0.0 比大小，負數就會自動變成 0.0
    overlap_w = np.maximum(0.0,overlap_w)
    overlap_h = np.maximum(0.0,overlap_h)
    
    # 把配對各自的 (重疊寬 * 重疊高) 乘起來，再用 np.sum 全部加總
    overlap_area = np.sum(overlap_w*overlap_h)
    
    # new總成本公式
    total_energy = (ALPHA * box_area) + (BETA * wire_cost) + (OVERLAP_PENALTY * overlap_area)
    return float(total_energy)

if __name__ == "__main__":
    # 測試已知最佳解
    best_R = np.array([[0.0, 10.0,10.0,10.0], [10.0, 0.0,10.0,10.0], [10.0, 10.0,0.0,0.0], [10.0, 10.0,10.0,10.0]])
    print("已知最佳解能量 :", energy(best_R))

    # 測試隨機初始解與移動鄰近解
    r0 = initial_solution()
    print("隨機初始解 R0:\n", r0)
    print("初始能量 E(R0):", energy(r0))

    r1 = neighbor(r0, 10.0)
    print("移動後鄰近解 R1:\n", r1)
    print("移動後能量 E(R1):", energy(r1))