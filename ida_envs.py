import numpy as np


##################################
#    NAT HUM BEH:: EXP 1
##################################
def get_T(envstep):
    NUM_STATES = len(envstep)
    T=np.zeros([NUM_STATES, NUM_STATES])
    T[0,2]=1
    T[2,4]=1
    T[1,3]=1
    T[3,5]=1
    return T
#####################
def getSRT(T,gamma):
    I = np.identity(len(T))
    SRT = np.linalg.inv(I - gamma * T )
    return SRT
#####################
def generate_nathumbeh_env1():
    NUM_STATES = 6
    nA = 1

    # envstep[s]=[  S_prime, reward , , Done]
    envstep=[]
    for s in range(NUM_STATES):
        envstep.append([0,0,False])
    
    envstep[0]=[2,0, False]
    envstep[2]=[4, 0,False]

    envstep[1]=[3, 0,False]
    envstep[3]=[5, 0,False]

    # TERMINAL STATES BELOW
    envstep[4]=[4,100, True]
    envstep[5]=[5,10, True]
    return envstep

def plot_env1():
    envstep = generate_nathumbeh_env1()
    #for i in [0,5]:
    from graphviz import Digraph

    g = Digraph('G', filename='exp1.gv')

    g.edge('1', '3')
    g.edge('3', '5')
    #g.edge('Bistro1', '100 reward')
    g.edge('2', '4')
    g.edge('4', '6')
    #g.edge('6', '7')
    #g.edge('7', 'Bistro2')
    #g.edge('Bistro2', '120 reward')
    g.view()
    
##################################
#       NAT COMM BISTRO
##################################
def generate_bistro():

    NUM_STATES = 8
    nA = 1

    # envstep[s]=[  S_prime, reward , , Done]
    envstep=[]
    for s in range(NUM_STATES):
        envstep.append([0,0,False])

    # 0 -> 2 -> 4 100$    
    envstep[0]=[2,0, False]
    envstep[2]=[4, 0,False]
    envstep[4]=[4,100, True] # TERMINAL STATE

    # 1 -> 3 -> 5 -> 6 -> 7 120$
    envstep[1]=[3, 0,False]
    envstep[3]=[5, 0,False]
    envstep[5]=[6,0, False]
    envstep[6]=[7,0, False]
    envstep[7]=[7,130, True] # TERMINAL STATE
    return envstep

##################################


