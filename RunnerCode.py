#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:07:51 2019

Project File for Intro. to AI

@author: prajwalrauniyar
"""

import numpy as np
import matplotlib.pyplot as pt
from copy import deepcopy as Dcp
from itertools import product

rows, cols = 20, 20

#Define the GOAL state here
goal = (rows-1, cols-1)
goal_reward = 7 #rows*cols*3

#Maximum cash that kalki can have when he spawns
max_cash = rows + cols
min_cash = 10
kalki = None #Object place holder

#Number of Asurs
n_asur = rows
asurs = []
asur_coordinates = [] #For easy lookup
MU_asur = None   #mu for reward
STDDEV_asur = 2 #stddev for reward

#Number of holes in the board
n_holes = 30
hole_coordinates = []

#The reward for moving to a regular cell
reg_call_reward = -0.7 

#Number of iterations we want the algo to run
num_iter = 2000

#Q learning params
Q = np.zeros((rows, cols))
choice_counts = np.zeros((rows, cols))
visit_counts = np.zeros((rows, cols))
all_rewards = []
gamma = 0.5
alpha = 0.1

#Finally set which algo to run, if set to True, this will run 
#Q-Learning algo
Q_LEARNING = True
#Set this to True if you want to see plot of Qvalues or rewards
#depending on the algorithm
plot_Q = True

class Kalki:
    def __init__(self,pos_x=None, pos_y=None, cash=None):
        self.x = pos_x if pos_x else np.random.randint(0,rows)
        self.y = pos_y if pos_y else np.random.randint(0,cols)
        self.cash = cash if cash else \
                    np.random.randint(min_cash, max_cash+1)
                    
    def get_position(self):
        return (self.x, self.y)
    
    def set_position(self,tup):
        self.x = tup[0]
        self.y = tup[1]
        
    def decrement_cash(self, howmuch=None):
        '''
        Decrements cash by 1 if howmuch is not set
        else decrements by howmuch
        '''
        self.cash = self.cash - 1 if howmuch == None else self.cash - howmuch
        
    

class Asur:
    def __init__(self,pos_x=None, pos_y=None, nature=None,
                 mu=None, stdev=None):
        self.mu = mu if not mu == None else 0
        self.stdev = stdev if not stdev==None else 1 
        self.x = pos_x if pos_x else np.random.randint(0,rows)
        self.y = pos_y if pos_y else np.random.randint(0,cols)
        
        #This value represents how evil the Asur is.
        #+ve value means the Asur is Good -ve means the Asur
        # evil
        #the magnitude depits the nature high -ve magnitude means
        # kalki will be killed, high +ve value means he gets more
        # cash
        self.nature = np.random.normal if not nature else nature
        
    def get_position(self):
        return (self.x,self.y)
    
    def get_reward(self):
        '''
        Depending on how evil the Asur is it will reward
        our hero. This number will be used to inflict punishment 
        or cash reward to our hero
        
        New cash can be cash = cash *(1 + an_asur.get_reward())
        '''
        p = self.nature(self.mu, self.stdev)
        return p #np.e**p/(1+np.e**p) - 0.4



def plot_state(state):
    """
    state is a dict as so
    state = {'kalki': (x,y),
             'asurs': [(x1,y1) , (x2,y2) ....],
             'holes': [(x1,y1), (x2,y2)....]
             }
    """
    pt.clf()
    if not isinstance(state, dict):
        raise Exception('State is not a dict')
    pt.grid()
    #Draw Kalaki
    pt.scatter(state['kalki'][0], state['kalki'][1], s=60, c='b')
    
    #Draw Asurs
    for asr in state['asurs']:
        pt.scatter(asr[0], asr[1], s=35, c='r')
        
    #Draw Holes
    for hl in state['holes']:
        pt.scatter(hl[0], hl[1], s=35, c='black')
        
    pt.xlim([-1, rows+1])
    pt.ylim([-1, cols+1])
    
def plot_learn_values(data):
    '''
    data must be a 2d which are usually the reward or
    Q values to be plotted
    '''
    pt.clf()
    
    pt.plot(np.arange(data.flatten().shape[0]),
            data.flatten())
    pt.xlabel('States (flattened)')
    pt.ylabel('Learnt values')
    pt.show
    
def initialize_board(do_or_not=True):
    '''
    Creates the initial board and places the asurs and charcters
    in their place
    
    This is just default 
    '''
    if not do_or_not:
        #If this is set we do not initialize. Assumption is that
        #these variables are already defined and need not be initialized
        return
    #global variable kalki
    global kalki
    global asurs
    global asur_coordinates
    global hole_coordinates
    global Q
    global choice_counts
    global MU_asur
    global STDDEV_asur
    
    
    hole_coordinates = [(np.random.randint(0,rows), (np.random.randint(0,cols)))
                        for _ in range(n_holes)]
    kalki = Kalki()
    while kalki.get_position() in hole_coordinates:
        kalki = Kalki()

    asurs = [Asur(mu=MU_asur, stdev=STDDEV_asur) for _ in range(n_asur)]
    asur_coordinates = [(i.x,i.y) for i in asurs]
    
    for each_hole in hole_coordinates:
        Q[each_hole[0], each_hole[1]] = -1
    
def construct_state():
    '''
    Using the global variables this fun creates as state of the game
    that can be plotted
    '''
    global kalki
    global asurs
    global asur_coordinates
    global hole_coordinates
    
    a_state = {}
    a_state['kalki'] = kalki.get_position()
    a_state['asurs'] = [i.get_position() for i in asurs]
    a_state['holes'] = Dcp(hole_coordinates)
    
    return a_state
    
def get_possible_states(tup):
    '''
    get possible states that our hero can goto from
    params 'x' and 'y'
    Also considers if holes are nearby
    
    Returns a list of possible coordinates
    '''
    x = tup[0]
    y = tup[1]
    return list(
            filter(
                    lambda a: (a not in hole_coordinates+[(x,y)]) and 
                                (a[0] < rows and a[0] >=0) and
                                (a[1] < cols and a[1] >=0)  ,
                    product([x-1,x,x+1],[y-1,y,y+1])
            )
            )


while goal in asur_coordinates:
    goal = get_possible_states(goal)[ 
                np.random.randint(0, get_possible_states(goal)) 
                                    ]

#Initstates and plot
initialize_board()
plot_state( construct_state() )

if Q_LEARNING:
    reached = False

    #Begin running here
    for i in range(num_iter):
        if reached:
            break
        curr_xy = kalki.get_position() 
        visit_counts[ curr_xy ] += 1
        explore = (np.random.uniform(0,1) < visit_counts[ curr_xy ]**-1)
        if explore:
            next_state = np.random.randint( np.array(
                            get_possible_states( curr_xy ) 
                                    ).shape[0] 
                            )
            next_state = get_possible_states( curr_xy )[next_state]
        else:
            next_state = np.argmax( 
                    [ Q[i] for i in get_possible_states( curr_xy ) ] 
                    )
        
            next_state = get_possible_states( curr_xy )[next_state]
        
        choice_counts[ next_state ] += 1
    
        #Set Kalki to be in next_state
        kalki.set_position(next_state)
    
        if next_state == goal:
            a_reward = goal_reward
        elif next_state in asur_coordinates:
            a_reward = asurs[ asur_coordinates.index( next_state ) ].get_reward()
        else:
            a_reward = reg_call_reward 
    
        all_rewards.append(a_reward)
    
        alpha = 1./choice_counts[next_state]
        Q[curr_xy] = (1-alpha)*Q[curr_xy] + alpha*(a_reward + gamma*Q[next_state])
    
        if i%10 == 0:
            #print('\n\nITER: %s' %i)
            plot_state( construct_state() )
            pt.show()
            if plot_Q:
                plot_learn_values(Q)
            # pt.pause(1.)
            pt.pause(.001)
            
        if kalki.get_position() == goal:
            reached = True
     
    print('Number of Iteration Ran: %s' %i)
    if reached:
        print('Kalki reached GOAL!!! WIth a cumilative reward of %s\nHe now has a total of:%s' %(sum(all_rewards), sum(all_rewards)+ kalki.cash ))   
    else:
        print('Kalki Failed in his Quest, Could collect total of %s\nHe now has a total of:%s' %(sum(all_rewards), sum(all_rewards)+ kalki.cash))
    #per step reward -0.7 , Asur.stdev=2 , 
    
#############################################
#TREE algo below

if not Q_LEARNING:
    visited = np.zeros((rows, cols))
    rewards = np.zeros((rows, cols))

    path = []

    rewards = rewards + reg_call_reward
    rewards[goal] = goal_reward
    for i in hole_coordinates:
        rewards[i] = -1 * reg_call_reward * 3  #We dont want hero to look for nay rewards
        visited[i] = 1                    #in holes nor do want to visit holes
    
    def traverse(current, visited, rewards):
        global plot_Q
        
        kalki.set_position( current )
        plot_state( construct_state() )
        pt.show()
        
        if plot_Q:  #Set it globally
            plot_learn_values(rewards)
            pt.show()
            
        if visited[current] == 1:    #Already visited node, return just the reward
            return rewards[current]  #since we have visited this node already
        
        visited[current] = 1
        if current == goal:
            return goal_reward
        
        allowed = get_possible_states( current )
        allowed = list ( filter( lambda x: visited[x] == 0 , 
                         get_possible_states(current) ) )
    
        if allowed == []: #Check if nowhere else to explore
            if current in asur_coordinates:
                indx = asur_coordinates.index( current )
                rewards[current] = asurs[indx].get_reward()
                return rewards[current]
            else:
                return rewards[current]
        else:
            allowed_vals = list( map(lambda x: traverse(x, visited, rewards),
                                    allowed ) )
            the_highest = np.argmax( allowed_vals )
            path.append( (allowed[the_highest],allowed_vals[the_highest] ) ) #Put the one selcted with max retuens in the path
        
            if current in asur_coordinates:
                indx = asur_coordinates.index( current )
                rewards[current] = asurs[indx].get_reward()
            #rerurn Reward from max child + self reward
            rewards[current] = allowed_vals[the_highest] + rewards[current]
            return rewards[current]
        
    complete_reward = traverse(kalki.get_position(), visited, rewards)
    print('After looking at all the states, Reward for Kalki is: %s' %complete_reward)
    

    

    
    
    
    
    
        
    
 

















