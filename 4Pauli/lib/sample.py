import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.convert import Convert

class Sample:
    
    def __init__(self, N) :
        
        if not os.path.exists("./data/N"+str(N)+"/sample/") :
            os.makedirs("./data/N"+str(N)+"/sample/")
        
        self.N = N
        self.csv = pd.read_csv("./data/csv/"+str(N)+"qubit.csv")
        
        self.lm = 3**N
        self.lc = len(self.csv.loc[0])-2
        
        return
    
    def sample(self, Ns, POVM) :
        
        print("N = {:3s}  Ns = {:5s}  POVM = {:6s}".format(str(self.N), str(Ns), POVM))
        
        print("Labels and prob...")
        prob = []
        prob_dic = {}
        for it in range(6**self.N) :
            la = self.csv["s0"][it][2:]
            
            self.N_M = 6
            if POVM == "4Pauli" :
                self.N_M = 4
                
                la = la.replace("1", "m")
                la = la.replace("-", "m")
                la = la.replace("r", "m")

            if la in prob_dic :
                prob_dic[la] += sum(self.csv.loc[it].values[2:])
            else :
                prob_dic[la] = sum(self.csv.loc[it].values[2:])
                
        labels = []
        prob = []
        for it in prob_dic :
            labels.append(it)
            prob.append(prob_dic[it])
            
        prob = np.array(prob) / sum(prob)
        
        np.save("./data/N"+str(self.N)+"/sample/labels.npy", labels)
        np.save("./data/N"+str(self.N)+"/sample/prob.npy", prob)
        
        conv = Convert(self.N, POVM)
        samp = []
        samP = np.zeros(self.N_M**self.N)
        print("Sampling...")
        for it in range(Ns) :
            m = 2**self.N * np.random.randint(0, self.lm, 1)[0]
            ind = np.random.randint(1, self.lc+1, 1)[0]

            value = self.csv["s"+str(ind)][m:m+2**self.N].values
            arg = np.where(value==1)[0][0]
            
            la = self.csv["s0"].loc[m + arg][2:]
            if POVM == "4Pauli" :
                la = la.replace("1", "m")
                la = la.replace("-", "m")
                la = la.replace("r", "m")
                
            samp.append(conv.label2state(la))
            samP[conv.label2num(la)] += 1
            
        np.save("./data/N"+str(self.N)+"/sample/samp.npy", samp)
        np.save("./data/N"+str(self.N)+"/sample/samP.npy", samP/sum(samP))
        
        print("Sample Done.")

        
        return 
        
        
        