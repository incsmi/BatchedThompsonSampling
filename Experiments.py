import numpy as np
import matplotlib.pyplot as plt
import timeit

def ER(mean,T,repeat,K,alpha,TS_var,reward_dist,std,sty):
    time=np.arange(1,T+1,1)
    #lists to hold the experiment data
    Regret=np.zeros((len(alpha),repeat,T)) 
    BatchCount=np.zeros((len(alpha),repeat,T))

    regret_TS=np.zeros((repeat,T))

    start_whole=timeit.default_timer()
    for i in range(repeat):
        start = timeit.default_timer()
        result=TSExp(T,mean,K,alpha,TS_var,reward_dist,std)
        result_regret=result[0]
        result_batch=result[1]
        stop = timeit.default_timer()
        print('Experiment no',i+1,"time:", stop - start,"seconds")
        for j in range(len(alpha)):
            Regret[j,i,:]=result_regret[j,:]
            BatchCount[j,i,:]=result_batch[j,:]

        regret_TS[i]=result[2]



    stop_whole = timeit.default_timer()
    print('Total Experiment Duration:', (stop_whole - start_whole)/60,"minutes")

    #averaging the data
    Regret_mn=np.zeros((len(alpha),T))
    BatchCount_mn=np.zeros((len(alpha),T))

    for i in range(len(alpha)):
        Regret_mn[i]=np.mean(Regret[i,:,:],axis=0)
        BatchCount_mn[i]=np.mean(BatchCount[i,:,:],axis=0)                 

    regret_TS_mn=np.mean(regret_TS,axis=0)

    #Plotting the results
    for i in range(len(alpha)):
        plt.plot(np.log10(time),Regret_mn[i],sty[i],label="Batched TS, alpha= "+str(alpha[i])+" ("+str(int(np.ceil(BatchCount_mn[i,-1])))+" batches)")
    plt.plot(np.log10(time),regret_TS_mn,color="green",linestyle=(0, (3, 1, 1, 1, 1, 1)),label="Normal TS"+" ("+str(T)+" batches)")
    plt.xlabel('log10(T)')
    plt.ylabel('The Average R(T)')
    plt.legend()
    plt.show()

    pass

def ActionSel(A_rew,A_sel,no):
    K=len(A_rew)
    action_sel=np.zeros((K,no))
    for i in range(K):
        var=1.0/A_sel[i]
        mn=A_rew[i]
        std=np.sqrt(var)
        action_sel[i]=mn+std*np.random.standard_normal(no)
    #selecting the most reward returning arm from the samples
    sel=np.argmax(action_sel,axis=0)
    return sel

def TSExp(T,mean,K,alpha,TS_var,reward_dist,std):
    #recorded rewards
    Rew1=np.zeros((len(alpha),K))
    Rew2=np.zeros((len(alpha),K))
    
    D_rew=np.zeros(K)
    
    #recorded selection count
    Sel1=np.ones((len(alpha),K))
    Sel2=np.ones((len(alpha),K))
    
    D_sel=np.ones(K)
    
    #cycle count
    Cycle=np.zeros((len(alpha),K))
    
    #statistics
    Limit=np.ones((len(alpha),K))
    R=np.zeros(len(alpha))
    Prev=[-1]*len(alpha)
    Regret=np.zeros((len(alpha),T))
    BatchCount=np.zeros((len(alpha),T))
    BatchCurr=[1]*len(alpha)
    CycleCount=[0]*len(alpha)
    CycleCheck=[0]*len(alpha)
    Mem=[0]*len(alpha)
    

    r_d=0
    regret_d=np.zeros(T)

    rewards=np.zeros(K)
    for t in range(T):
        #reward generation
        if reward_dist=="B":
            for i in range(K):
                rewards[i]=np.random.binomial(1,mean[i],1)
        elif reward_dist=="P":
            rewards=np.random.poisson(mean)
        elif reward_dist=="G":
            rewards=mean+std*np.random.standard_normal(K)
        else:
            raise ValueError("Error: Wrong reward_dist")

        for i in range(len(alpha)):
            Sel=ActionSel(Rew1[i,:],Sel1[i,:]/TS_var[i],1)[0]
            R[i]+=mean[0]-mean[Sel]
            Regret[i,t]=R[i]
            BatchCount[i,t]=BatchCurr[i]


            Sel2[i,Sel]+=1
            Rew2[i,Sel]=(rewards[Sel]/Sel2[i,Sel])+((Sel2[i,Sel]-1)/(Sel2[i,Sel]))*Rew2[i,Sel]
        
            if Sel!=Prev[i] or CycleCheck[i]==0:
                Mem[i]=Prev[i]
                Cycle[i,Sel]+=1
                Prev[i]=Sel
                CycleCheck[i]+=1
                if CycleCheck[i]==2:
                    CycleCheck[i]=0
                    if Limit[i,Mem[i]]==Cycle[i,Mem[i]] or Limit[i,Prev[i]]==Cycle[i,Prev[i]]:
                        BatchCurr[i]+=1
                        for j in range(K):
                            if Cycle[i,j]>0:
                                Limit[i,j]=int(np.ceil(alpha[i]*Cycle[i,j]))
                            Rew1[i,j]=Rew2[i,j]
                            Sel1[i,j]=Sel2[i,j]
        

        sel_d=ActionSel(D_rew,D_sel/TS_var[-1],1)[0]
        r_d+=mean[0]-mean[sel_d]
        regret_d[t]=r_d
        D_sel[sel_d]+=1
        D_rew[sel_d]=(rewards[sel_d]/D_sel[sel_d])+((D_sel[sel_d]-1)/(D_sel[sel_d]))*D_rew[sel_d]

    return [Regret,BatchCount,regret_d]

#LIST OF PARAMETERS

#No of actions
K=2

#Batch Growth Factor
alpha=[2,1.5,1.25,1.00001]

#TS sampling variance, sigma^2: each variance component corresponds to a alpha value respectively plus the normal Thompson sampling at the end (-1 index)
TS_var=np.ones(len(alpha)+1)

#Graph properties: modify line styles for each alpha value (make sure to set this to the same length as alpha vector)
sty=["k-","b--","m:","r-."]

#reward dist: "G" for Gaussian, "B" for Bernoulli, "P" for Poisson
reward_dist="G"

#Gaussian noise std: works only for Gaussian distribution
std=1

#expected mean for each arm: set its length to K and make sure the 0 index has the biggest value
mean=np.zeros(K)+0.25
mean[0]=0.75

#Experiment Duration
T=5*10**4

#Experiment repeat count
repeat=10000

#EXPERIMENT FUNCTION

ER(mean,T,repeat,K,alpha,TS_var,reward_dist,std,sty)
    
