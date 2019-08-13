import numpy as np
import matplotlib.pyplot as plt
import random
from SR_Dyna_no_action import SR_Dyna_no_action
#from dyna_replay import dyna_replay

def SRDyna_nathum_exp1(envstep, gamma, alpha, p_sample=None, verbose=0):

    ''' This function uses the reinfrocement learning agent class in 
        SR_no_action.py to learn.
        Here the function takes the environment from Experiment 1 in our
        Nat Hum Beh paper & learns predictive representations with the 
        specified learning rate and scale. 

        Note: This is an SR dyna agent (), not SR-MB.         

        Inputs: 
        
        envstep:  generated with ida_envs.generate_nathumbeh_env1()        
        gamma: discount parameter, determines scale of predictive representations
        alpha: learning rate
        p_sample: prob sampling each of the two sequences 
        verbose: 0: none, 1: very verbose, 2: a bit

        Outputs:

        M: SR matrix 
        W: value weights W
        memory: memory of episodes 
        episodies: # episodes it takes to reach convergence 

        Ida Momennejad, 2019'''

    if p_sample==None:
        p_sample= [.5,.5]

    SR_agent = SR_Dyna_no_action(gamma, alpha, p_sample, len(envstep))
    episodes = 0
    done = False
   
    while True:
        SR_agent.biggest_change = 0
        # sample a starting point [really determined by experiment]
        s = np.random.choice(range(2), p=SR_agent.p_sample)

        done = False                
        while not done: # go through trajectory till the end
            
            s_new, reward, done = envstep[s] #take action        
            SR_agent.step(s, s_new, reward)
            
            s = s_new
        
        if verbose==2:            
            print(f'SR training episode #{episodes} Done.')       
        episodes += 1        

        if SR_agent.convergence:
            if verbose==2: 
                print (episodes,' training episodes/iterations done')
            break

        else: 
            ''' SR DYNA PART '''
            SR_agent.dyna_replay()
            if verbose==2:  
                if episodes % 10 ==0:            
                    print(f'SRDYNA episode #{episodes} Done.')

    return SR_agent.M, SR_agent.W , SR_agent.memory, episodes


