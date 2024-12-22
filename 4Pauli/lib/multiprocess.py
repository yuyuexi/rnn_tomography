import os, time
import numpy as np
import cvxpy as cp
from scipy import linalg as lg
import matplotlib.pyplot as plt
from multiprocessing import Pool


def kron(args) :
    
    name, Md = args[0], args[1]

    res = np.array([1])
    for it in name :
        res = np.kron(res, Md[it])
                
    return res

def Kron(label, Md) :
    
    args = [[it, Md] for it in label]
        
    pool = Pool()
    res = pool.map(kron, args)
    
    pool.close()
    pool.join()
    
    return res
    
def mle(args) :
    
    p, N, M = args[0], args[1], args[2]
    dm = cp.Variable([2**N, 2**N], complex=True)
    error = [cp.trace(it@dm) for it in M] - p
        
    constraints = [dm == dm.H, cp.trace(dm)==1, dm>>0]
    objective = cp.Minimize(cp.norm(cp.hstack(error)))
    problem = cp.Problem(objective, constraints)
    
    problem.solve(solver=cp.SCS)
    
    res = np.array([np.trace(it.dot(dm.value)) for it in M])
        
    return res

def MLE(pm, N, M) :
    
    args = [[it, N, M] for it in pm]
    
    pool = Pool(processes=10)
    res = pool.map(mle, args)
    
    pool.close()
    pool.join()
    
    return res

def cfidelity(args) :
    
    p1, p2 = args[0], args[1]
    
    res = np.sum(np.power(p1*p2, 0.5))
    
    return res

def CFidelity(p1, p2) :
    
    args = [[p1, it] for it in p2]
    
    pool = Pool()
    res = pool.map(cfidelity, args)
    
    pool.close()
    pool.join()
    
    return res

def qufidelity(args) :
    
    s1, p2, unit = args[0], args[1], args[2]
    
    s2 = np.einsum("a,aij->ij", p2, unit)
    plt.matshow(s2.real)

    sqr = lg.sqrtm(s1)
    inner = np.einsum("ij,jk,kl", sqr, s2, sqr)
    
    a, b = np.linalg.eig(inner)
    
    res = sum(np.sqrt(a[a>=0]))
    
    return res

def QuFidelity(s1, p2, unit) :
    
    args = [[s1, it, unit] for it in p2]
    
    pool = Pool()
    res = pool.map(qufidelity, args)
    
    pool.close()
    pool.join()
    
    return res


def qucorr(args) :
    
    p2, unit, Sigma = args[0], args[1], args[2]
    
    s2 = np.einsum("a,aij->ij", p2, unit)
    
    res = np.einsum("ij,ji", s2, Sigma)
    
    return res

def QuCorr(p2, unit, Sigma) :
    
    args = [[it, unit, Sigma] for it in p2]
    
    pool = Pool()
    res = pool.map(qucorr, args)
    
    
    pool.close()
    pool.join()
    
    return res

    
    