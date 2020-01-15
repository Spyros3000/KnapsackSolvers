from ortools.algorithms import pywrapknapsack_solver
import time
import pandas as pd

tic = time.perf_counter()

dataframe= pd.DataFrame(columns=['Problems', 'Time_OR-Tools','Time_Solver-CPython','Total_value','Total_weight'])

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

solver = pywrapknapsack_solver.KnapsackSolver(
          pywrapknapsack_solver.KnapsackSolver
              .KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
          'Dynamic Programming solver')

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
    solver.Init(value, weight, capacities)
    start1 = time.perf_counter()
    computed_value = solver.Solve()
    end1 = time.perf_counter()
    start2 = time.perf_counter()
    simpleSolverSolution=knapSack(capacities[0],templist,value,int(lineList[0]))
    end2 = time.perf_counter()
	# Necessary for OR Tools
    packed_items = []
    packed_weights = []
    total_weight = 0
    for i in range(len(value)):
        if solver.BestSolutionContains(i):
            packed_items.append(items[i])
            packed_weights.append(weight[0][i])
            total_weight += weight[0][i]
	# Save to Pandas dataframe
    dataframe= dataframe.append({'Problems': str(n) + "_" + str(r) + "_" + str(t) + "_" + str(z) + '_5', 'Time_OR-Tools': end1 - start1,'Time_Solver-CPython':end2-start2,'Total_value':computed_value,'Total_weight':total_weight},
                   ignore_index=True)

n_table=[10,50,100,500]
r_table=[50,100,500,1000]

for n in n_table:
    for r in r_table:
        for t in range(1,5,1):
            for z in range(1,6,1):
                mainSolver(n,r,t,z)
dataframe.to_csv("resultsCPython.csv",index=False)

toc = time.perf_counter()
print('Ran in ' + str(toc - tic) + ' seconds.')

import matplotlib.pyplot as plt
ax = plt.gca()
ax.set_title('Solving times in different solutions')
ax.set_xlabel('Problems')
ax.set_ylabel('Time (seconds)')
ax.legend(['OR-Tools','CPython','Pypy'])
dataframe.plot(x='Problems',y=['Time_OR-Tools','Time_Solver-CPython'],ax=ax,kind='line')

plt.show()
