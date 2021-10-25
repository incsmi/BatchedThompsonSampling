# BatchedThompsonSampling
Instructions to run the experiments:

Executable file: Experiments.py
Code language: Python 3
Libraries needed: numpy, matplotlib, timeit

All the relevant parameters are given at the bottom of the code. 
We have described each parameter and how they should be initialized in relation to each other there.
You can edit those parameters using any text editor, and double click on Experiments.py to run the experiment.

The current code is configured to Figure 1.a in Section 5, and we present specifications for each figure:

1.a)
K=2
alpha=[2,1.5,1.25,1.00001]
TS_var=np.ones(len(alpha)+1)
sty=["k-","b--","m:","r-."]
reward_dist="B"
std=1
mean=np.zeros(K)+0.25
mean[0]=0.75
T=5*10**4
repeat=10000

1.b)
K=5
alpha=[2,1.5,1.25,1.00001]
TS_var=np.ones(len(alpha)+1)
sty=["k-","b--","m:","r-."]
reward_dist="B"
std=1
mean=np.zeros(K)+0.25
mean[0]=0.75
T=5*10**4
repeat=10000


1.c)
K=2
alpha=[2,1.5,1.25,1.00001]
TS_var=np.ones(len(alpha)+1)
sty=["k-","b--","m:","r-."]
reward_dist="G"
std=1
mean=np.zeros(K)
mean[0]=1
T=5*10**4
repeat=10000

1.d)
K=5
alpha=[2,1.5,1.25,1.00001]
TS_var=np.ones(len(alpha)+1)
sty=["k-","b--","m:","r-."]
reward_dist="G"
std=1
mean=np.zeros(K)
mean[0]=1
T=5*10**4
repeat=10000


