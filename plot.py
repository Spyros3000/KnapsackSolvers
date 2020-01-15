import pandas as pd

dataframePypy = pd.read_csv('resultsPypy.csv')
dataframeCPython = pd.read_csv('resultsCPython.csv')

import matplotlib.pyplot as plt
ax = plt.gca()
ax.set_title('Solving times in different solutions')
ax.set_xlabel('Problems')
ax.set_ylabel('Time (seconds)')
ax.legend(['OR-Tools','CPython','Pypy'])
dataframePypy.plot(x='Problems',y='Time_Solver-Pypy',ax=ax,kind='line')
dataframeCPython.plot(ax=ax, x='Problems', y=['Time_OR-Tools','Time_Solver-CPython'])

plt.show()
