#!/usr/bin/env python
# coding: utf-8



import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import random
from bisect import bisect_right 
import time


# N: total population, M: number of edge nodes
# P:testing period, Q: number of nodes being tested
def network_iter(G,Q,P,N=4000,M=800,b=0.1,gamma=0.2): 
    N_nodes = set(np.arange(0,N,1))             
    # randomly sample Q testing subjects from whole population
    test_subjects = random.sample(N_nodes,Q)             
    # randomly sample M edges nodes (boundary conditions) from whole population
    edges = random.sample(N_nodes,M)
    # randomly select 1 node from M edge nodes to be first infected
    infected= random.sample(set(edges),1)             
    # rest are susceptible 
    suscep=[s for s in N_nodes if s not in infected]     
    
    removed=[]

    for inf in infected:                           #initialize node attribute
        G.nodes[inf]['status']='infected'
        G.nodes[inf]['inf_dur']= 0
        G.nodes[inf]['ID']= 'None'

    for sus in suscep:
        G.nodes[sus]['status']='susceptible'
        G.nodes[sus]['ID']= 'None'
        
    for t in test_subjects:
        G.nodes[t]['ID']= 'Tested'
        
            
    counter = 0
   
    gamma_inverse = 1/gamma
    I = 1
    S = N-I
    R = 0
    dt = 1

    I_record=[]
    S_record=[]
    R_record=[]
    TotalCases = I+R
    finished = False

    while len(infected)>0 and finished==False:

        new_infected = []

        for sus in suscep:
            nei = G[sus].keys()       # get current susceptible's neighbors
            # get infectious neighbors
            infected_nei = [n for n in nei if G.nodes[n]['status']=='infected']
            p_infection = 1-np.power((1-b*dt),len(infected_nei))
            inf_status = np.random.binomial(1,p_infection)
            if inf_status==1:
                new_infected.append(sus)
                I=I+1
                S=S-1

        new_removed = []

        for inf in infected:

            G.nodes[inf]['inf_dur']=G.nodes[inf]['inf_dur']+dt

            if G.nodes[inf]['inf_dur']>=gamma_inverse:
                new_removed.append(inf)
                I=I-1
                R=R+1


        new_infected= list(dict.fromkeys(new_infected))
        new_removed = list(dict.fromkeys(new_removed))

        for re in new_removed:
            infected.remove(re)
            G.nodes[re]['status']='removed'

        for inf in new_infected:
            suscep.remove(inf)
            G.nodes[inf]['status']='infected'
            G.nodes[inf]['inf_dur']=0

        infected.extend(new_infected)
        removed.extend(new_removed)
        counter = counter + 1
        

        I_record.append(len(infected))
        R_record.append(len(removed))
        S_record.append(len(suscep))
        
        if(counter%P==0):
            for t in test_subjects:
                if (G.nodes[t]['status']=='infected' or G.nodes[t]['status']=='removed'):
                    finished = True
                    break
            
                    
    TotalCases = len(infected)+len(removed)        
    
    return S_record,I_record,R_record,TotalCases


def test(num_lines=5, max_t=30):
    N = 4000
    Q = 40
    P = 100000
    G = nx.barabasi_albert_graph(N, 3)

    lines_shape = (num_lines, max_t+1)
    S_lines = np.empty(lines_shape)
    I_lines = np.empty(lines_shape)
    R_lines = np.empty(lines_shape)

    for i in range(num_lines):
        S, I, R, _ = network_iter(G, Q, P, N)
        S, I, R = np.array(S), np.array(I), np.array(R)
        S_lines[i, :S.shape[0]] = S[:max_t+1]
        I_lines[i, :I.shape[0]] = I[:max_t+1]
        R_lines[i, :R.shape[0]] = R[:max_t+1]

    fig = plt.figure()
    S_mean = np.nanmean(np.array(S_lines), axis=0)
    I_mean = np.nanmean(np.array(I_lines), axis=0)
    R_mean = np.nanmean(np.array(R_lines), axis=0)

    #plt.plot(S_mean, label="S", color=u'#1f77b4', linewidth=2)
    plt.plot(I_mean, label="I", color=u'#ff7f0e', linewidth=2)
    #plt.plot(R_mean, label="R", color=u'#2ca02c', linewidth=2)

    for I in I_lines:
        plt.plot(I, color=u'#ff7f0e', linewidth=0.5)

    plt.legend()
    plt.grid(which="major")
    plt.xlim(0, max_t)
    plt.show()
    fig.savefig("SIR_curves_old.png")


random.seed(123)
np.random.seed(123)
test()

