## a one-time python script to generate reference-target image pairs for the specific-designed dataset of WHU aerial image.
## Each scene contains (40+2)*15+30=660 images, where there are 20 columns and 10 rows. Each two rows have the reverse order. So the odd rows will go from left to right while the even rows will reverse. So the structure is a little bit complicated.

import numpy as np 

output_file = "pairs.txt"

names = np.zeros((16,40),dtype="object")
count = 0
for row in range(16):
    if row%2 == 0:
        column_range = range(40)
    else:
        column_range = range(39,-1,-1)
    for column in column_range:
        names[row][column] = f"{count:04}.png"
        count += 1
        if count == 660:
            break
    count += 2

pairs_head = {}
f = open(output_file,"w")

for row in [0]:
    for col in range(1,39):
        ref_target = [names[row][col],names[row][col-1],names[row][col+1],"xxxx.png",names[row+1][col]]
        pairs_head[int(ref_target[0][:-4])] = ref_target
for row in [15]:    
    for col in range(38,10,-1):
        ref_target = [names[row][col],names[row][col-1],names[row][col+1],names[row-1][col],"xxxx.png"]
        pairs_head[int(ref_target[0][:-4])] = ref_target
for row in range(1,15):
    for col in range(1,39):
        ref_target = [names[row][col],names[row][col-1],names[row][col+1],names[row-1][col],names[row+1][col]]
        ref_target[4] = "xxxx.png" if ref_target[4] == 0 else ref_target[4]
        pairs_head[int(ref_target[0][:-4])] = ref_target

for row in range(0,15):
    idx_group = [(row+1)*42-i for i in range(4)]
    if row%2 == 0:
        pairs_head[idx_group[0]] = [f"{idx_group[0]:04}.png", names[row+1][38], "xxxx.png", f"{idx_group[1]:04}.png", names[row+2][39] if row<14 else "xxxx.png"]
        pairs_head[idx_group[1]] = [f"{idx_group[1]:04}.png", names[row+1][38], "xxxx.png", f"{idx_group[2]:04}.png", f"{idx_group[0]:04}.png"]
        pairs_head[idx_group[2]] = [f"{idx_group[2]:04}.png", names[row][38], "xxxx.png", f"{idx_group[3]:04}.png", f"{idx_group[1]:04}.png"]
        pairs_head[idx_group[3]] = [f"{idx_group[3]:04}.png", names[row][38], "xxxx.png", names[row-1][39] if row>0 else "xxxx.png", f"{idx_group[2]:04}.png"]
    else:
        pairs_head[idx_group[0]] = [f"{idx_group[0]:04}.png", "xxxx.png", names[row+1][1], f"{idx_group[1]:04}.png", names[row+2][1] if row<13 else "xxxx.png"]
        pairs_head[idx_group[1]] = [f"{idx_group[1]:04}.png", "xxxx.png", names[row+1][1], f"{idx_group[2]:04}.png", f"{idx_group[0]:04}.png"]
        pairs_head[idx_group[2]] = [f"{idx_group[2]:04}.png", "xxxx.png", names[row][1], f"{idx_group[3]:04}.png", f"{idx_group[1]:04}.png"]
        pairs_head[idx_group[3]] = [f"{idx_group[3]:04}.png", "xxxx.png", names[row][1], names[row-1][0], f"{idx_group[2]:04}.png"]

pairs_head[0] = [names[0][0], "xxxx.png", names[0][1], "xxxx.png", "0081.png"]
pairs_head[659] = [names[15][10], "xxxx.png", names[15][11], names[14][10], "xxxx.png"]

for i in range(660):
    if i in pairs_head:
        f.write(f"{pairs_head[i][0]} {pairs_head[i][1]} {pairs_head[i][2]} {pairs_head[i][3]} {pairs_head[i][4]}\n")

f.close()
