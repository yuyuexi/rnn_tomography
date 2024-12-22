import numpy as np

class Convert:
    
    def __init__(self, N, POVM) :
        
        
        self.N = N
        try :
            self.labels = np.load("./data/N"+str(N)+"/sample/labels.npy")
        except :
            print("No labels found.")
            return 
        
        if POVM == "6Pauli" :
            self.N_M = 6
            self.table = np.array(["0", "+", "l", "1", "-", "r"])
        if POVM == "4Pauli" :
            self.N_M = 4
            self.table = np.array(["0", "+", "l", "m"])
            
        return
    
    def label2ind(self, label) :
        ind = []
        for it in label :
            ind.append(np.where(self.table==it)[0][0])
            
        return np.array(ind)
    
    def ind2label(self, ind):
        label = ""
        for it in ind :
            label += self.table[it]
            
        return label
    
    def ind2state(self, ind) :
        state = np.array([])
        for it in ind :
            one = np.eye(self.N_M)[it]
            state = np.concatenate([state, one])
            
        return state
    
    def state2ind(self, state) :
        ind = []
        for it in np.reshape(state, [-1, self.N_M]):
            ind.append(np.argmax(it))
            
        return np.array(ind)
    
    def label2state(self, label) :
        
        ind = self.label2ind(label)
        state = self.ind2state(ind)
        
        return state
    
    def num2label(self, num) : 
        
        try:
            assert(0<= num < self.N_M**self.N)
        except:
            print("Num should be set in range(0, N_M^N).")
            return -1
        
        return self.labels[num]
    
    def label2num(self, label) :
        
        return np.where(self.labels==label)[0][0]
    
    def num2state(self, num) :
        
        try:
            assert(0<= num < self.N_M**self.N)
        except:
            print("Num should be set in range(0, N_M^N).")
            return -1
        
        label = self.num2label(num)
        state = self.label2state(label)
        
        return state
    
    def state2num(self, state) :
        
        ind = self.state2ind(state)
        label = self.ind2label(ind)
        num = self.label2num(label)
        
        return num
    
    
    

        
            

        
            