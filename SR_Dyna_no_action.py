import numpy as np
import matplotlib.pyplot as plt
import random
import math

class SR_Dyna_no_action():

	''' This class defines a reinforcement learning agent that 
	learns the state-state successor representation without taking actions. 
	Thus, the resulting SR matrix is in the service of prediction. 

	Initalization parameters

	gamma: discount param
	alpha: learning rate
	p_sample: probability of sampling different options, only relevant for testing poilcy dependence
	NUM_STATES: the number of states in the environment to intialize matrices

	Ida Momennejad, 2019'''

	def __init__(self, gamma, alpha, p_sample, NUM_STATES):
		self.gamma = gamma # discount factor
		self.replay_gamma = gamma # can be different in future
		self.alpha = alpha # learning rate
		self.replay_alpha = alpha # can be different in future

		self.p_sample = p_sample # p(sampling options)		

		# Initialize the successor representation, M with I
		self.M= np.eye(NUM_STATES) # M: state-state SR    	
		#self.M= np.zeros([NUM_STATES, NUM_STATES]) # M: state-state SR    	

		self.W= np.zeros([NUM_STATES]) # W: value weights, 1D
		self.onehot=np.eye(NUM_STATES) # onehot matrix, for updating M
		self.n = NUM_STATES
		
		self.V= np.zeros([NUM_STATES]) # value function
		self.biggest_change = 0
		self.significant_improvement = 0.001 # convergence threshold
		# policy: not revelant in exp 1, agent is passively moved
		#		  but in Exp2 we keep updating it to get the optimal policy
    	# self.Pi = np.zeros([NUM_STATES], dtype=int)  
		self.epsilon = .1
		self.memory=[]

	def onehot_row(self, successor_s):	
		row = np.zeros( len(self.W)) 
		row[successor_s] = 1
		return row

	def step(self, s, s_new, reward):

		old_v = self.get_value()

		self.update_memory(s, s_new) # memory for replay
		self.update_SR(s, s_new)
		self.update_W(s, s_new, reward)

		self.update_biggest_change(old_v[s], s)

        ########## update policy  ##############
        #Pi[s] = action
        # M, W = dyna_replay(memory, M, W, episodes)


	def update_SR(self, s, s_new):

		# onehot here with s_new vs. with s
		onehot_row = self.onehot_row(s_new )
		SR_TD_error = onehot_row + self.gamma * self.M[s_new] -self.M[s]  
		#print(f's {s} to {s_new}: {SR_TD_error}')


		# learning by element, as opposed to by row
		self.M[s, s_new]   =  self.M[s,s_new] + .2*SR_TD_error[s_new]

		# Lines below are 2 row-update approaches 
		#self.M[s]   =  self.M[s] + self.alpha * SR_TD_error
		#self.M[s] = (1-self.alpha)* self.M[s] + self.alpha * ( self.onehot[s] + self.gamma * self.M[s_new]  )		
		#self.M[s] = (1-self.alpha)* self.M[s] + self.alpha * ( self.onehot[s_new] + self.gamma * self.M[s_new]  )

		

	def update_W(self, s, s_new, reward):

		''' Update value weight vector. 
		It computes the normalized feature vector * reward PE.
        Here reward function would be sufficient. The same, 
        but R is easier. We use W in plos comp biol 2017 paper, to 
        account for BG weights allowing dopamine similarities 
        between  MF and MB learning.'''

		# future notes: 27 feb 2019: in paper both get updated with every transition
		# better to do batch updates. W updated every step, but M 
		# updated every couple of steps with dyna
		# like feature learning.
		# all rules are correct, but in practice for TD learning on features
		# a little weird to learn feature vector with every step
		# normally features are stable over the task.

		norm_feature_rep = self.M[s] / ( self.M[s]@self.M[s].T ) 

		# Compute the values of s and s_prime, then the prediction error

		V_snew = self.M[s_new]@self.W  
		V_s    = self.M[s]@self.W 		                          
		w_pe = ( reward + self.gamma*V_snew - V_s ).squeeze()        

		# Update W with the same learning rate
		# future: this could be different
		self.W += self.alpha * w_pe *norm_feature_rep
        
	def get_value(self):
		''' Combine the successor representation M & value weight W
			to determine the value of different options'''

		self.V = self.M@self.W		
		return self.V

	def update_memory(self, s, s_new):
		''' Save current state and the state it visited in one-step
			to memory. This is used in the Dyna version for replay.'''

		self.memory.append([s, s_new])

	def dyna_replay(self, replay_steps = 1000, nsamples = 50):
		# default: 1000, 50. change when caled
		
		# let's confine the memory to replay_steps steps
		# for recency. 
		# XX CAN CHANGE IN FUTURE, not like ploscomp XX
		#print(replay_steps)

		if len(self.memory)<replay_steps:
			mem=self.memory
		else:
			mem=self.memory[-replay_steps: -1]

		# get exponentially weighted replay samples
		replay_samples = self.get_replay_samples(mem, nsamples)

		# replay and learn from samples:
		for sample in replay_samples:
			rands, next_s = self.memory[sample] # get next state from memory
			MPE = self.onehot[rands]+self.replay_gamma*self.M[next_s]
			# ** update successor representation (not W)
			self.M[rands] = (1-self.replay_alpha)*self.M[rands]+self.replay_alpha*MPE

	def get_replay_samples(self, mem, nsamples):	
		''' Takes the memory bag & number of sapmles, returns
		** exponential recency weighted ** replay samples

		input: mem, samples
		output: replay_samples

		** NOTE: in prioritized sweeping Dyna samples 
		are drawn based on relation to priority queue --> |PE|
		'''
		
		# 1) compute recency weighted probabilities 
		y=[] 
		for i in range(len(mem)):
			y.append(math.exp(i/25)) # division: more chance to smaller numbers, avoids range error
		norm_p_weights = [float(i)/sum(y) for i in y]

		# 2) sample with recency & replacement
		replay_samples = np.random.choice(len(mem), nsamples, p=norm_p_weights, replace=True)
		return replay_samples

	def update_biggest_change(self, old_v_m, s):
		''' Coomputes the change in value, see if it is higher
			than the present max change, if so, update biggest_change '''

		V=self.get_value()
		self.biggest_change = max(self.biggest_change, np.abs(old_v_m - V[s]))   
		self.check_converegnce()         
	
	def check_converegnce(self):
		''' If statement is true, conferegnce has reached. '''

		self.convergence= self.biggest_change < self.significant_improvement
		
    	
            