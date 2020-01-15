import time
import pandas as pd

tic = time.perf_counter()

dataframe= pd.DataFrame(columns=['Problems','Time_Solver-Pypy'])


def knapSack(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    return K[n][W]


def mainSolver(n,r,t,z):
	# Read files, use lineList to handle them
    fileHandle = open('problems/problem_'+str(n)+'_'+str(r)+'_'+str(t)+'_'+str(z)+'_'+'5.txt', "r")
    lineList = fileHandle.readlines()
    fileHandle.close()
    value = []
    weight = []
    capacities = []
    items = []
	# Templist is needed in order to create a 2D table, as OR Tools expects
    templist=[]
    global dataframe
    capacities.append(int(lineList[-1]))
    for g in range(1, len(lineList) - 1):
        splitter = lineList[g].split()
        value.append(int(splitter[1]))
        items.append(splitter[0])
        templist.append(int(splitter[2]))
    weight.append(templist)
    start=time.time()
    simpleSolverSolution=knapSack(capacities[0],templist,value,int(lineList[0]))
    end = time.time()

	# Save to Pandas dataframe
    dataframe= dataframe.append({'Problems': str(n) + "_" + str(r) + "_" + str(t) + "_" + str(z) + '_5', 'Time_Solver-Pypy':end-start},ignore_index=True)

n_table=[10,50,100,500]
r_table=[50,100,500,1000]

for n in n_table:
    for r in r_table:
        for t in range(1,5,1):
            for z in range(1,6,1):
                mainSolver(n,r,t,z)
dataframe.to_csv("resultsPypy.csv",index=False)

toc = time.perf_counter()
print('Ran in ' + str(toc - tic) + ' seconds.')
