import os, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from lib.convert import Convert


class RNN :
    
    def __init__(self, POVM, N=4, train_steps=200, batch_size=100, G=1, gru=5, latent=10):
        
        self.N = N
        self.POVM = POVM
        self.conv = Convert(N, POVM)
        
        if POVM == "4Pauli":
            self.N_M = 4
        if POVM == "6Pauli":
            self.N_M = 6
            
        self.path = "./data/N" + str(N)
        
        self.train_steps = train_steps
        self.batchsize = batch_size
        
        self.G = G
        self.gru = gru
        self.latent = latent
        
        cons = np.array([self.conv.num2state(num) for num in range(self.N_M**self.N)])
        self.cons_np = np.reshape(cons,[self.N_M**self.N,self.N,self.N_M])
        self.cons = tf.constant(self.cons_np,dtype=tf.float32)
        
        return 
    
    def loss_func(self, samp):
        
        self.data = tf.reshape(samp,[self.batchsize,self.N,self.N_M],name="data")

        gen_data = self.generation(self.data)
        con_p = self.generation(self.cons)

        self.loss = tf.reduce_mean(-tf.reduce_sum(self.data*tf.log(1e-10+gen_data),[1,2]),name="loss")
        
        logP = tf.reduce_sum(self.cons*tf.log(1e-30+con_p),[1,2])
        self.prob = tf.exp(logP)

        return self.loss
    
    def generation(self,data=None):
        
        batchsize = tf.shape(data)[0]
        
        with tf.variable_scope("generation", reuse=tf.AUTO_REUSE):
            
#             latent =  tf.get_variable('latent', shape=[1, self.latent])
#             blatent_ = tf.tile(latent, [batchsize, 1])
#             blatent = tf.layers.dense(inputs=blatent_, units=self.latent, activation=tf.nn.relu)

            ins = tf.slice(tf.concat([tf.zeros([batchsize,1,self.N_M],dtype=tf.float32),data],axis=1),[0,0,0],[batchsize, self.N, self.N_M])
#             inputs = tf.stack([tf.concat([it, blatent], axis=1) for it in tf.unstack(ins, axis=1)], axis=1)
            
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.gru) for _ in range(self.G)], state_is_tuple=False) 
            
            outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=ins, dtype=tf.float32) 
            
            dense = tf.reshape(outputs, [batchsize*self.N, self.gru])
            dlayer = tf.layers.dense(dense, units=self.N_M, activation=tf.nn.softmax)
            
            denseh = tf.reshape(dlayer, [batchsize, self.N, self.N_M])
            
        return denseh

    def train(self):
        
        train_data = np.load(self.path+"/sample/samp.npy")
        
        batch_hold = tf.placeholder(dtype=tf.float32,shape=[self.batchsize,self.N*self.N_M])
        loss_func = self.loss_func(samp=batch_hold)
        
        global_step=tf.Variable(0,trainable=False)
        lr=tf.train.exponential_decay(0.01,global_step,10,0.5,staircase=True)

        optimizer = tf.train.AdamOptimizer(lr).minimize(loss_func)
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        
        prob = []
        print("N = {:7s}   Ns = {:5s}       POVM: {:7s}".format(str(self.N), str(train_data.shape[0]), str(self.POVM)))
        with  tf.Session() as sess:
            sess.run(init)
                    
            start = time.time()
            for epoch in range(self.train_steps):
                bcount = 0
                train_data = np.random.permutation(train_data)
                while bcount * self.batchsize < train_data.shape[0]:
                    if bcount * self.batchsize + self.batchsize <= train_data.shape[0]:
                        batch_data = train_data[bcount * self.batchsize: bcount * self.batchsize + self.batchsize]
                  
                    ls = [optimizer,self.loss,self.prob]
                    _, loss, p = sess.run(ls,feed_dict={batch_hold:batch_data})
            
                    bcount += 1
                
                prob.append(p)
                    
                if epoch % 10 == 0 :
                    print("Epoch = {:4s}  Loss = {:.4f}    Time = {:.4f} ".format(str(epoch), loss, time.time()-start))

        self.save_result(prob)
        print("Training Done.")
        
        return 
    
    def save_result(self, prob):
        
        path = self.path + "/rnn/"
        if not os.path.exists(path):
            os.makedirs(path)
            
        np.save(path + "prob.npy", prob)
        
        return 
    
    