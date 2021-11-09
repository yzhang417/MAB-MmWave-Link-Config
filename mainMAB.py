#----------------------------------------------
# Import packages
#----------------------------------------------
import argparse
import os
import os.path
from os import path
import sys
import math
import random
import time
import pdb
import operator 
import copy
import pickle
import platform
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib as mb
import torch
from envs import *
from state_action import *
from get_MAB_policy_setting import *
from decision_making import *
from my_utils import *
from pprint import pprint
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'int': '{:2d}'.format})
np.set_printoptions(formatter={'float': '{:6.3f}'.format})

#-----------------------------------------------------------
# main function
#-----------------------------------------------------------
def main():
    print('\r\n------------------------------------')
    print('Enviroment Sumamry')
    print('------------------------------------')
    print('PyTorch ' + str(torch.__version__))
    print('Running with Python ' + str(platform.python_version()))    
    
    #-----------------------------------------------------------
    # Parse command line arguments
    #-----------------------------------------------------------
    print('\r\n------------------------------------')
    print('System Parameters')
    print('------------------------------------')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Mode
    parser.add_argument('--output', default=None, help='output folder of training results')
    # Training process
    parser.add_argument('--slots', default=1000, type=int, help='number of slots in a single episode')
    parser.add_argument('--LOOP', default=500, type=int, help='number of episodes')
    parser.add_argument('--exid', default=5, type=int, help='Example ID')
    
    #Print args
    args = parser.parse_args()
    for item in vars(args):
        print(item + ': ' + str(getattr(args, item)))
    
    #----------------------------------------------
    # Initialization
    #----------------------------------------------
    slots_monitored = args.slots;         # Total time slots to be monitored in testing
    LOOP = args.LOOP; # Number of realization
    exid = args.exid;
    env_parameter = env_init()
    t_slot_min = min(env_parameter.t_slot_vec)
    t_slot_max = max(env_parameter.t_slot_vec)
    np.set_printoptions(suppress=True)
    #np.set_printoptions(precision=2)
    
    #-----------------------------------------------------------
    # Check output folder
    #-----------------------------------------------------------
    if args.output is not None:
        output_folder = args.output+'/'
    else:
        output_folder = 'output/'        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print('Minimum time monitored is %0.2f seconds' %(slots_monitored*t_slot_min))
    print('Maximum time monitored is %0.2f seconds' %(slots_monitored*t_slot_max))
    print('Maximum activity range is %d meters' %env_parameter.max_activity_range)
    print('Average time in blockage is \r\n', env_parameter.target_prob_blockage)
    print('Probabbility of blockage is \r\n', env_parameter.prob_blockage)

    #----------------------------------------------
    # All MAB algorithms settings
    #----------------------------------------------
    policy_setting_list = get_MAB_policy_setting(exid,env_parameter)
    
    #----------------------------------------------
    # Result vectors
    #----------------------------------------------
    N_polices = len(policy_setting_list)
    evolution_reward = np.zeros((LOOP,slots_monitored,N_polices)); # Evolution of reward of the whole training process
    evolution_bw = np.zeros((LOOP,slots_monitored,N_polices)); # Evolution of reward of the whole training process
    evolution_tslot = np.zeros((LOOP,slots_monitored,N_polices)); # Evolution of reward of the whole training process
       
    #-----------------------------------------------------------
    # Random seed
    #-----------------------------------------------------------
    #random.seed(13579)     # random seeds for reproducation
    #np.random.seed(246810) # random seeds for reproducation
    random.seed()     # random seeds
    np.random.seed() # random seeds
        
    #----------------------------------------------
    # Training MAB-based scheduler
    #----------------------------------------------
    # Timer start
    tStart = time.time()
    
    # LOOP of realizations
    for loop in range(LOOP):
        # Default state/action/output/banditBW/banditRelay list
        state_list = list();
        action_list = list();
        env_list = list();
        for policy_id in range(N_polices):
            env_parameter = env_init(policy_setting_list[policy_id].ratio_codebook_used)
            env_list.append(envs(env_parameter,slots_monitored));
            state_list.append(def_state(env_list[policy_id].env_parameter));
            action_list.append(def_action(env_list[policy_id].env_parameter)); 
            
        # Random moving for 200 time slots
#         for ct in range(slots_monitored):
#             for policy_id in range(N_polices):
#                 reward, MCS_level = env_list[policy_id].step(state_list[policy_id], action_list[policy_id]);
#         for policy_id in range(N_polices): 
#             env_list[policy_id].ct = 0;
        
        # Loop of slots in the training process
        for ct in range(slots_monitored):
            # Loop of polices (one-step interaction with enviroment and update)
            for policy_id in range(N_polices):
            #for policy_id in [4]:
                # Decisiion making
                action = decision_making(state_list[policy_id], env_list[policy_id].env_parameter, exid, policy_setting_list[policy_id], ct);                
                # Interaction with enviroment
                reward, MCS_level = env_list[policy_id].step(state_list[policy_id], action);                
                # Step-wise bandit vector update
                if ct > 0:
                    state_list[policy_id] = update_bandit_para(env_list[policy_id].env_parameter, state_list[policy_id], action, reward, MCS_level, exid, policy_setting_list[policy_id], ct);
                    # Save result
                    if exid == 4 and policy_setting_list[policy_id].enable_bandit:
                        evolution_reward[loop,ct,policy_id] = state_list[policy_id].new_reward; # Evolution of reward of the whole training process
                    else:
                        evolution_reward[loop,ct,policy_id] = reward; # Evolution of reward of the whole training process
                    evolution_bw[loop,ct,policy_id] = action.bw_id; # Evolution of reward of the whole training process
                    evolution_tslot[loop,ct,policy_id] = action.tslot_id; # Evolution of reward of the whole training process
    
    elapsed_time = time.time() - tStart
    print('\nExecution time per loop is %0.2f seconds' %(elapsed_time/LOOP))
    print('\n')
    
    # Plot network topology    
    show_plot = False
    show_mobility_trace = False;
    #plot_network_topology(env_list[0].env_parameter, output_folder, show_mobility_trace, show_plot, exid) # plot network topology
    show_mobility_trace = True;
    plot_network_topology(env_list[0].env_parameter, output_folder, show_mobility_trace, show_plot, exid) # plot network topology
    
    # Plot evolution of reward (real data rate)
    plot_result(env_list[0].env_parameter, evolution_reward, evolution_tslot, policy_setting_list, exid, output_folder)
    
    # Save all variable
    args_MAB = args
    training_results_filename = output_folder+'example'+str(exid)+'.pt'
    training_results_dict = {
        'args_MAB': args_MAB,
        'policy_setting_list': policy_setting_list,
        'env_parameter': env_list[0].env_parameter,
        'evolution_reward': evolution_reward,
        'evolution_bw': evolution_bw,
        'evolution_tslot': evolution_tslot
    }
    outfile = open(training_results_filename,'wb')
    pickle.dump(training_results_dict, outfile)
    outfile.close()

    
if __name__ == "__main__":
    main()


