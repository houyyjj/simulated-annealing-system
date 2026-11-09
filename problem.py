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


#初始定義

#晶片邊界長寬
BOUND_MIN = -100.0
BOUND_MAX = 100.0

#懲罰權重
OVERLAP_PENALTY = 10.0
#重疊1罰10

#IC模組清單(w寬,h高)
MODULES = [ (10.0,10.0),(10.0,10.0),(10.0,10.0),(10.0,10.0)]

# step1:產生初始解R0
def initial_solution():
    R0 = []
    for w,h in MODULES:
        max_x = BOUND_X - w
        max_y = BOUND_Y - h
        x = random.uniform(0.0,max_x)
        y = random.uniform(0.0,max_y)
        R0.append([x,y])
    return R0


# step3:由現解R0產生鄰近解R1
def neighbor(R,s):
    #COPY目前座標狀態
    R1 = copy.deepcopy(R)
    
    #隨機挑選第idx個模組
    idx = random.randint(0,len(R1) - 1)
    w,h = MODULES[idx]
    rand_x = random.random()
    rand_y = random.random()
    
    #R1 = (R0-s/2)+s*rand(1)
    new_x = R1[idx][0] - (s / 2.0) + s*rand_x
    new_y = R1[idx][1] - (s / 2.0) + s*rand_y
    
    #count不可超過晶片邊界
    max_x = BOUND_X - w
    max_y = BOUND_Y - h
    
    #check邊界
    if new_x <0.0:
        new_x = 0.0
    elif new_x > max_x:
        new_x = max_x
    
    if new_y <0.0:
        new_y = 0.0
    elif new_y > max_y:
        new_y = max_y
    
    R1[idx][0] = new_x
    R1[idx][1] = new_y
    return R1

# 能量 = 所有大外框面積+模組重疊懲罰
def energy(R):
    #找外框四個極端邊界
    #先拿第0個模組當起點
    min_x = R[0][0]
    max_x = R[0][0] + MODULES[0][0]
    min_y = R[0][1]
    max_y = R[0][1] + MODULES[0][1]
    
    #
    for i in range(len(R)):
        x = R[i][0]
        y = R[i][1]
        w,h = MODULES[i]
        
        if x<min_x:
            min_x = x