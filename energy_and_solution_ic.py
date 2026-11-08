import numpy as np 

# 1.基本參數設定
#晶片最大邊界長寬(100*100)
BOUND_X = 100.0
BOUND_Y = 100.0

#成本函數三個權重: Cost = ALPHA*Area + BETA*Σ(Wij*HPWLij) + GAMMA*Overlap
ALPHA = 1.0   #面積權重 α
BETA = 1.0    #線長延遲權重 β
GAMMA = 10.0  #重疊懲罰權重 γ (重疊1罰10)

#IC模組清單[w寬,h高] -> 4*2 (0:CPU, 1:GPU, 2:I/O, 3:Cache)
MODULES = np.array([[10.0,10.0],[10.0,10.0],[10.0,10.0],[10.0,10.0]])

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
    R0 = np.column_stack((x,y))
    #numpy 二維(4,2)，每列是一顆模組的[x,y]
    return R0

# 3: 由現解 R 產生鄰近解R1(step3:隨機移動一顆方塊)
def neighbor(R,s):
    
    R1 = R.copy()
    # 從0,1,2,3中隨機抽一個號碼idx決定這次要移動哪一顆模組
    idx = np.random.randint(0,len(R1))
    #取被抽取的模組w,h
    w,h = MODULES[idx]
    
    #各自產生0.0~1.0之間的隨機亂數 rand(1)
    rand_x = np.random.rand()
    rand_y = np.random.rand()
    
    #新座標 = 舊座標 - (擾動範圍s/2)+s*亂數
    #R1[idx,0] 第idx顆的x座標
    new_x = R1[idx,0] - (s/2.0) + s*rand_x
    new_y = R1[idx,1] - (s/2.0) + s*rand_y
    
    #方塊合法最大邊界
    max_x = BOUND_X - w
    max_y = BOUND_Y - h
    
    #np.clip(數值,最小值,最大值)->限制範圍
    R1[idx,0] = np.clip(new_x,0.0,max_x)
    R1[idx,1] = np.clip(new_y,0.0,max_y)
    
    return R1

# 4.count 能量函數 E(R) (step1&4:算排的好壞)
# 能量越低->排越好：Cost = α*Area + β*Σ(Wij*HPWLij) + γ*Overlap
def energy(R):
    #R[:, 0]->把全部4顆方塊的左邊界 x整排抓出來， np.min 找出最左邊的值；
    # 左邊界 x加上寬度 w=右邊界， np.max 找出最右邊的值
    min_x = np.min(R[:,0])
    max_x = np.max(R[:,0]+MODULES[:,0])
    # 下邊界
    min_y = np.min(R[:,1])
    max_y = np.max(R[:,1]+MODULES[:,1])
    
    # 1.大外框面積 Area = 總寬度(最右 - 最左) * 總高度(最上 - 最下)
    box_area = (max_x - min_x) * (max_y - min_y)
    
    #自動配對
    #n=4為例->i:[0,0,0,1,1,2] j:[1,2,3,2,3,3]
    n = len(R)
    i,j = np.triu_indices(n,k=1)
    
    #根據配對號碼i抓左邊方塊的x1y1w1h1
    x1,y1 = R[i,0],R[i,1]
    w1,h1 = MODULES[i,0],MODULES[i,1]
    #j，抓右邊方塊
    x2,y2 = R[j,0],R[j,1]
    w2,h2 = MODULES[j,0],MODULES[j,1]
    
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
    total_energy = (ALPHA * box_area) + (BETA * wire_cost) + (GAMMA * overlap_area)
    return float(total_energy)

if __name__ == "__main__":
    # 測試已知最佳解：4 個 10x10 排成 20x20 田字型 (面積400 + 線長180 + 重疊0 = 580.0)
    best_R = np.array([[0.0, 10.0], [10.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    print("已知最佳解能量 (預期 580.0):", energy(best_R))

    # 測試隨機初始解與移動鄰近解
    r0 = initial_solution()
    print("隨機初始解 R0:\n", r0)
    print("初始能量 E(R0):", energy(r0))

    r1 = neighbor(r0, 10.0)
    print("移動後鄰近解 R1:\n", r1)
    print("移動後能量 E(R1):", energy(r1))