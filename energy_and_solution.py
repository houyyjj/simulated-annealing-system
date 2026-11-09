'''
1. 定義能量函數(適應函數) E(R)，且於結構空間隨機產生一初始解R0。
2. 設定起始溫度 T 和降溫係數𝑑_𝑡及適當地設定k值，並以初始解當做現解。
3. 由現解 R0 的擾動範圍s產生一鄰近解
	 𝑅_1=(𝑅_0−𝑠/2)+𝑠×𝑟𝑎𝑛𝑑(1) 	 			
其中 rand(1)∈[0, 1]為一隨機所產生的亂數，並計算其能量函數值E(R1)。
4. 計算鄰近解之能量函數值和現解之能量函數值之差值 ∆E=E(𝑅_1 )−E(𝑅_0)若 ∆E≤0 ，則令 𝑅_0=𝑅_1 ；否則，產生一亂數 rand(1)∈[0, 1] ，假如
            rand(1)<Pr⁡(∆E)=exp((−∆𝐸)/𝑘𝑇)，則令 𝑅_0=𝑅_1 。 (註: Pr 為機率函數)
5. 重覆步驟3及步驟4  m 次，m為每個溫度下之疊代次數或稱為擾動次數。
6. 進行降溫，並同時減少其擾動範圍
	 	T=T×𝑑_(𝑡  );      𝑆=𝑆×𝑑_𝑠  				
式中， 𝑑_𝑡, 𝑑_𝑠∈[0,1]，𝑑_𝑠為擾動空間下降係數，其值最好在0.8 ~ 0.95之間。
'''
import numpy as np 

# 1.基本參數設定
# 搜尋空間最大邊界長寬(100*100)
BOUND_X = 100.0
BOUND_Y = 100.0

# 成本函數權重: Cost = ALPHA*Area + BETA*HPWL + OVERLAP_PENALTY*Overlap
ALPHA = 1.0            # 面積權重 α
BETA = 1.0             # 線長權重 β
OVERLAP_PENALTY = 10.0 # 重疊懲罰權重 γ (重疊1罰10)

# 基底預設參數：4個通用物件，每個邊長為 10.0
N = 4
SIZE = 10.0

# 2.產生初始解R0(step1)
def initial_solution():
    # 4個物件的x,y最大合法邊界
    max_x = BOUND_X - SIZE
    max_y = BOUND_Y - SIZE

    # 隨機在合法範圍內抽出4個x,y座標
    x = np.random.uniform(0.0, max_x, N)
    y = np.random.uniform(0.0, max_y, N)
    
    # 合併直排x,直排y
    R0 = np.column_stack((x, y))
    # numpy 二維(4,2)，每列是一個物件的[x,y]
    return np.array(R0)

# 3: 由現解 R 產生鄰近解R1(step3:隨機移動一個物件)
def neighbor(R, s):
    
    R1 = R.copy()
    # 從0,1,2,3中隨機抽一個號碼idx決定這次要移動哪一個
    idx = np.random.randint(0, len(R1))
    
    # 各自產生0.0~1.0之間的隨機亂數 rand(1)
    rand_x = np.random.rand()
    rand_y = np.random.rand()
    
    # 新座標 = 舊座標 - (擾動範圍s/2)+s*亂數
    new_x = R1[idx, 0] - (s / 2.0) + s * rand_x
    new_y = R1[idx, 1] - (s / 2.0) + s * rand_y
    
    # 合法最大邊界
    max_x = BOUND_X - SIZE
    max_y = BOUND_Y - SIZE
    
    # np.clip(數值,最小值,最大值)->限制範圍
    R1[idx, 0] = np.clip(new_x, 0.0, max_x)
    R1[idx, 1] = np.clip(new_y, 0.0, max_y)
    
    return R1

# 4.count 能量函數 E(R) (step1&4:算排的好壞)
# 能量越低->排越好：總能量 = α*外框面積 + β*總線長 + (重疊面積*OVERLAP_PENALTY)
def energy(R):
    # R[:, 0]->把全部左邊界 x整排抓出來，np.min 找出最左邊的值；
    # 左邊界 x加上邊長 SIZE=右邊界，np.max 找出最右邊的值
    min_x = np.min(R[:, 0])
    max_x = np.max(R[:, 0] + SIZE)
    # 下邊界與上邊界
    min_y = np.min(R[:, 1])
    max_y = np.max(R[:, 1] + SIZE)
    
    # 1.大外框面積 = 總寬度(最右 - 最左) * 總高度(最上 - 最下)
    box_area = (max_x - min_x) * (max_y - min_y)
    
    # 自動配對
    # n=4為例->i:[0,0,0,1,1,2] j:[1,2,3,2,3,3]
    n = len(R)
    i, j = np.triu_indices(n, k=1)
    
    # 根據配對號碼i抓左邊物件的x1,y1，j抓右邊物件的x2,y2
    x1, y1 = R[i, 0], R[i, 1]
    x2, y2 = R[j, 0], R[j, 1]
    
    # 2.count HPWL線長：兩物件中心點的 寬差(|cx1-cx2|) + 高差(|cy1-cy2|)
    cx1, cy1 = x1 + (SIZE / 2.0), y1 + (SIZE / 2.0)
    cx2, cy2 = x2 + (SIZE / 2.0), y2 + (SIZE / 2.0)
    hpwl = np.abs(cx1 - cx2) + np.abs(cy1 - cy2)
    wire_cost = np.sum(hpwl)
    
    # 3.count重疊面積：小右界np.minimum(右1,右2)-大左界np.maximum(左1,左2)
    overlap_w = np.minimum(x1 + SIZE, x2 + SIZE) - np.maximum(x1, x2)
    overlap_h = np.minimum(y1 + SIZE, y2 + SIZE) - np.maximum(y1, y2)
    
    # 如果沒重疊，上面減出來會是負數；用 np.maximum 跟 0.0 比大小，負數就會自動變成 0.0
    overlap_w = np.maximum(0.0, overlap_w)
    overlap_h = np.maximum(0.0, overlap_h)
    
    # 把配對各自的 (重疊寬 * 重疊高) 乘起來，再用 np.sum 全部加總
    overlap_area = np.sum(overlap_w * overlap_h)
    
    # 簡報P.19公式：面積 + 線長 + 重疊懲罰
    total_energy = (ALPHA * box_area) + (BETA * wire_cost) + (overlap_area * OVERLAP_PENALTY)
    return float(total_energy)

if __name__ == "__main__":
    # 測試已知最佳解：4 個 10x10 排成 20x20 田字型，完全不重疊 (面積400 + 線長80 + 重疊0 = 480.0)
    best_R = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    print("已知最佳解能量 (預期 480.0):", energy(best_R))

    # 測試隨機初始解與移動鄰近解
    r0 = initial_solution()
    print("隨機初始解 R0:\n", r0)
    print("初始能量 E(R0):", energy(r0))

    r1 = neighbor(r0, 10.0)
    print("移動後鄰近解 R1:\n", r1)
    print("移動後能量 E(R1):", energy(r1))