import numpy as np
import matplotlib.pyplot as plt
import random

def SR_nathum_exp1(envstep, gamma, alpha):

    #alpha = .5
    epsilon = .1


    NUM_STATES = len(envstep)
    # The Value for each state
    V = np.zeros([NUM_STATES]) 
    # Our policy, which we keep updating to get the optimal policy
    Pi = np.zeros([NUM_STATES], dtype=int)  
    ### M: the successor representation 
    M=np.zeros([NUM_STATES, NUM_STATES])
    ### W: the value weights, 1D
    W=np.zeros([NUM_STATES])
    onehot=np.eye(NUM_STATES)

    memory=[]
           



    significant_improvement = 0.0001
    episodes = 0

    ############ BEGIN ###############
    while True:
        # biggest_change is referred to by the mathematical symbol delta in equations
        biggest_change, biggest_change_M = 0, 0
        for s in range (0, NUM_STATES): # for every state
            
            ###############################################
            old_v = V[s]
            old_v_m = (M@W)[s]
            ###############################################
            # goto the state, take action 
            #env.env.s = s 
            s_new, rew, done = envstep[s] #take the action        
            ###############################################
            #update memory
            memory.append([s, s_new])
            ###############################################
            ## UPDATE M, W  
            M[s] = (1-alpha)* M[s] + onehot[s] + alpha *  gamma * M[s_new] #  here alpha==1
            
            ## UPDATE W: HERE"S WHERE REWARD COMES IN
            norm_feature_rep = M[s] / ( M[s]@M[s].T )                    
            
            w_pe = ( rew + gamma*(M[s_new]@W) - (M[s]@W)).squeeze()        
            W += alpha * w_pe *norm_feature_rep  
                            
            ########## update policy  ##############
            #Pi[s] = action
             
            ###############################################
            # Get biggest change in SR based Value function
            V = M@W 
            biggest_change_M = max(biggest_change_M, np.abs(old_v_m - V[s]))            

       
        #if episodes % 10 ==0:            
            #print(f'SRDYNA online & replay training episode #{episodes} Done.')       
        episodes += 1
        
        if biggest_change_M < significant_improvement:   # stop when not getting better!
            print (episodes,' training episodes/iterations done')
            break

    return M, W 