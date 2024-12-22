import os, time
import numpy as np
import cvxpy as cp
from scipy import linalg as lg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool

from lib.multiprocess import kron, Kron
from lib.multiprocess import mle, MLE
from lib.multiprocess import cfidelity, CFidelity
from lib.multiprocess import qufidelity, QuFidelity
from lib.multiprocess import qucorr, QuCorr


class Estimation:
    
    def __init__(self, N, POVM, Re=False, ty="rnn") :
        
        self.N = N
        self.POVM = POVM
        
        s0 = np.array([1,  0])
        s1 = np.array([0,  1])
        sa = np.array([1,  1]) / np.math.sqrt(2)
        sm = np.array([1, -1]) / np.math.sqrt(2)
        sr = np.array([1, 1j]) / np.math.sqrt(2)
        sL = np.array([1,-1j]) / np.math.sqrt(2)
        
        if POVM == "6Pauli" :
            self.N_M = 6
            self.M = np.zeros([self.N_M,2,2], dtype=np.complex)
            self.M[0] = np.einsum("i,j->ij", s0, s0)
            self.M[1] = np.einsum("i,j->ij", sa, sa)
            self.M[2] = np.einsum("i,j->ij", sL, np.conj(sL))
            self.M[3] = np.einsum("i,j->ij", s1, s1) 
            self.M[4] = np.einsum("i,j->ij", sm, sm) 
            self.M[5] = np.einsum("i,j->ij", sr, np.conj(sr))         
            self.M = self.M / 3
            self.Md = {"0":self.M[0], "+":self.M[1], "l":self.M[2], "1":self.M[3], "-":self.M[4], "r":self.M[5]}
            self.table = np.array(["0", "+", "l", "1", "-", "r"])
            
        if POVM == "4Pauli" :
            self.N_M = 4
            self.M = np.zeros([self.N_M,2,2], dtype=np.complex)
            self.M[0] = np.einsum("i,j->ij", s0, s0)
            self.M[1] = np.einsum("i,j->ij", sa, sa)
            self.M[2] = np.einsum("i,j->ij", sL, np.conj(sL))
            self.M[3] = np.einsum("i,j->ij", s1, s1) + np.einsum("i,j->ij", sm, sm) + np.einsum("i,j->ij", sr, np.conj(sr))         
            self.M = self.M / 3
            self.Md = {"0":self.M[0], "+":self.M[1], "l":self.M[2], "m":self.M[3]}
            self.table = np.array(["0", "+", "l", "m"])
        
        if POVM != "6Pauli" :
            print("Kron Producting...")
            label = np.load("./data/N"+str(N)+"/sample/labels.npy", allow_pickle=True)
            self.outM = np.array(Kron(label, self.Md))
            
            print("Inversing...")
            self.T = np.einsum("aij,bji->ab", self.outM, self.outM)
            self.Ti = np.linalg.inv(self.T)
            self.unit = np.einsum("ab,bij->aij", self.Ti, self.outM)
            
        

        print("Bayes...")
        if POVM == "6Pauli" :
            bm = np.load("./data/csv/bayesmatrix.npy", allow_pickle=True)[self.N-1]
        else :
            bm = np.eye(2**self.N)

        bm = np.eye(2**self.N)

        if ty == "rnn" :
            self.path = "./data/N" + str(self.N) + "/estimation/rnn/"
        if ty == "sam" :
            self.path = "./data/N" + str(self.N) + "/estimation/sam/"
        if not os.path.exists(self.path) :
            os.makedirs(self.path)

        pe = np.load("./data/N"+str(self.N)+"/sample/prob.npy", allow_pickle=True)
        pb = np.einsum("ij,jk->ik", bm, pe.reshape([2**self.N,-1])).reshape([1,-1])[0]
        np.save(self.path + "bayes.npy", pb)
        
        if ty == "rnn" :
            pm = np.load("./data/N" + str(self.N) + "/rnn/prob.npy", allow_pickle=True)[:100]
        if ty == "sam" :
            pm = np.load("./data/N" + str(self.N) + "/sample/samP.npy", allow_pickle=True)
            pm = np.array([pm for _ in range(10)])
        
          
        pmb = np.array([np.einsum("ij,jk->ik", bm, it.reshape([2**self.N,-1])).reshape([1,-1])[0] for it in pm])
        
        np.save(self.path + "rnnbayes.npy", pmb)
            

        
        
        if Re == True :
 
            
            print("Maximum Likelihood...")
            s = time.time()
            
            pmle = mle([pb, self.N, self.outM])
            np.save(self.path + "prob.npy", pmle)
            
            mlep = MLE(pmb, self.N, self.outM)
            np.save(self.path + "mlep.npy", mlep)
            
      
            print("MLE Time: {:25f}".format(time.time()-s))

            
        return 
    
    
    
    def fidelity(self, begin=0, end=-1, mle=False, cf=True, qf=False) :
        
        if self.POVM == "6Pauli" :
            qf = False
        
        self.QF = self.CF = None

        
        if mle == True :
            p1 = np.load(self.path+"prob.npy", allow_pickle=True)
            P2 = np.load(self.path+"mlep.npy", allow_pickle=True)[begin:end]
        else :
            p1 = np.load(self.path+"bayes.npy", allow_pickle=True)
            P2 = np.load(self.path+"rnnbayes.npy", allow_pickle=True)[begin:end]
        
        if cf == True :
            print("Classical Fidelity...")
            self.CF = CFidelity(p1, P2)
        
        if qf == True :
            print("Quantum Fidelity...")
            
            s1 = np.einsum("a,aij->ij", p1, self.unit)
            self.QF = QuFidelity(s1, P2, self.unit)
            
        if mle == True :
            title = "N="+str(self.N)+"   Fidelity with MLE"
        else :
            title = "N="+str(self.N)+"   Fidelity without MLE"
            
        fig = plt.figure(figsize=[12,3], dpi=300)
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(self.CF, label="Classical Fidelity="+str(max(self.CF).real)[:6])
        ax1.plot(len(self.CF)*[0.99], label="Control=0.9900")
        ax1.legend()
        
        if qf == False :
            ax1.set_title("N="+str(self.N)+"   Fidelity without MLE")
        
        if qf == True :
            ax2 = fig.add_subplot(1,2,2)
            ax2.plot(self.QF, label="Quantum Fidelity="+str(max(self.QF).real)[:6])
            ax2.plot(len(self.QF)*[0.9000], label="Control=0.9000")
            ax2.legend()
            fig.suptitle(title)
        
        plt.savefig(self.path+"fidelity.png", dpi=500)
        
            
            
        return self.CF, self.QF
    
    def correlation(self, m="z", begin=0, end=-1, mle=False, method=0) :
        
        if self.POVM == "6Pauli":
            method = 1
       
        
        def replace(num, i0, i1) :
            res = ""
            for it in "0"*(self.N-len(bin(num)[2:]))+bin(num)[2:] :
                if it == "0" :
                    res += i0
                if it == "1" :
                    res += i1
            return res
        
        
        if mle == True :
            p1 = np.load(self.path+"prob.npy", allow_pickle=True)
            P2 = np.load(self.path+"mlep.npy", allow_pickle=True)[begin:end]
        else :
            p1 = np.load(self.path+"bayes.npy", allow_pickle=True)
            P2 = np.load(self.path+"rnnbayes.npy", allow_pickle=True)[begin:end]
        
        print("Correlation...")
        if method == 0 :
            s1 = np.einsum("a,aij->ij", p1, self.unit)
            
            if m == "x" :
                sigma = np.array([[0,1],[1,0]])
            if m == "y" :
                sigma = np.array([[0,-1j],[1j,0]])
            if m == "z" :
                sigma = np.array([[1,0],[0,-1]])
            else :
                sigma = np.array([[1,0],[0,1]])
            
            Sigma = np.array([1])
            for it in range(self.N) :
                Sigma = np.kron(Sigma, sigma)
            
            c1 = np.einsum("ij,ji", s1, Sigma)
            C2 = QuCorr(P2, self.unit, Sigma)
            
        if method == 1 :
            try :
                assert self.POVM == "6Pauli"
            except :
                print("Mehtod 1 only for POVM = 6Pauli.")
                return 0, 0
            
            if m == "x" :
                i0 = "+"
                i1 = "-"
            if m == "y" :
                i0 = "l"
                i1 = "r"
            if m == "z" :
                i0 = "0"
                i1 = "1"
                
            arg = [replace(it, i0, i1) for it in range(2**self.N)]
            label = np.array(np.load("./data/N"+str(self.N)+"/sample/labels.npy", allow_pickle=True))
            ind = [np.where(label==it)[0][0] for it in arg]
            sign = np.array([(-1)**bin(it).count("1") for it in ind])
            c1 = sum(np.array(p1[ind])*sign)
            C2 = np.array([sum(np.array(it[ind])*sign) for it in P2])
            
        if mle == True :
            title = "N="+str(self.N)+"   Correlation with MLE"
        else :
            title = "N="+str(self.N)+"   Correlation without MLE"

        fig = plt.figure(figsize=[12,3], dpi=500)
        ax = fig.add_subplot(121)
        ax.plot(C2, label="RNN="+str(np.mean(C2[-10:]).real)[:6])
        ax.plot([c1]*len(C2), label="Experiment="+str(c1.real)[:6])
        ax.legend()
        ax.set_title(title)
        
        plt.savefig(self.path+"correlation.png", dpi=500)
        
        
            
        
        return c1, C2
    
    def image(self, indc=0, indq=0, MAX=True) :
        
        try :
            assert self.N < 6  
        except :
            print("Valid only for N<=5.")
            return 
        
        if MAX == True :
            indc = np.argmax(self.CF)
            indq = np.argmax(self.QF)
            
        fig = plt.figure(figsize=[20,30], dpi=500)
        
        ax1 = fig.add_subplot(3,2,1)
        ax2 = fig.add_subplot(3,2,2)
        
        p1 = np.load("./data/N"+str(self.N)+"/sample/prob.npy", allow_pickle=True)
        s1 = np.einsum("a,aij->ij", p1, self.unit)
        
        im1 = ax1.imshow(s1.real)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        ax1.set_title("Sample Real")
        ax1.set_xticks([]) 
        ax2.set_yticks([]) 
        
        im2 = ax2.imshow(s1.imag)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        ax2.set_title("Sample Imag")
        
        ax3 = fig.add_subplot(3,2,3)
        ax4 = fig.add_subplot(3,2,4)
        
        p2 = np.load("./data/N"+str(self.N)+"/rnn/prob.npy", allow_pickle=True)[indc]
        s2 = np.einsum("a,aij->ij", p2, self.unit)
        
        im3 = ax3.imshow(s2.real)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        ax3.set_title("CF Real")
        
        im4 = ax4.imshow(s2.imag)
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im4, cax=cax, orientation='vertical')
        ax4.set_title("CF Imag")
        
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)
        
        p3 = np.load("./data/N"+str(self.N)+"/rnn/prob.npy", allow_pickle=True)[indq]
        s3 = np.einsum("a,aij->ij", p3, self.unit)
        
        im5 = ax5.imshow(s3.real)
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im5, cax=cax, orientation='vertical')
        ax5.set_title("QF Real")

        im6 = ax6.imshow(s3.imag)
        divider = make_axes_locatable(ax6)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im6, cax=cax, orientation='vertical')
        ax6.set_title("QF Imag")
        
        plt.show()
        
        