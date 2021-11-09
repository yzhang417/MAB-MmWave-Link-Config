# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb


#-------------------------------------------------------------------------
# parameter for state
#-------------------------------------------------------------------------   
class def_state():
    def __init__(self, env_parameter):
        K_bw = env_parameter.K_bw;
        K_tslot = env_parameter.K_tslot;
        M = env_parameter.M;
        K_beam_max = max(env_parameter.N_SSW_BS_vec);
        
        # Example: 1. Beamwidth selection (UWMTS)
        self.UWMTS_Dir_par_bw = np.ones((K_bw,M+1));
        self.CountLeader_bw = np.zeros((K_bw,1));
        self.NumUse_bw = np.zeros((K_bw,1));
        self.Mean_bw = np.zeros((K_bw,1));
        self.Mean_bw[:] = math.inf;
        self.UCB_Index_bw = np.zeros((K_bw,1));
        self.UCB_Index_bw[:] = math.inf
        
        # Example: 2. Beam training periodicity selection (UWMTS)
        self.UWMTS_Dir_par_tslot = np.ones((K_tslot,M+1));
        self.CountLeader_tslot = np.zeros((K_tslot,1));
        self.NumUse_tslot = np.zeros((K_tslot,1));
        self.Mean_tslot = np.zeros((K_tslot,1));
        self.Mean_tslot[:] = math.inf;
        self.UCB_Index_tslot = np.zeros((K_tslot,1));
        self.UCB_Index_tslot[:] = math.inf
        
        # Example: 3. Joint Beamwidth and Beam training periodicity selection (KLUCB)
        self.GMTS_Dir_par_bw_tslot = np.ones((K_tslot*K_bw,M+1));  
        self.CountLeader_bw_tslot = np.zeros((K_tslot*K_bw,1));
        self.NumUse_bw_tslot = np.zeros((K_tslot*K_bw,1));
        self.Mean_bw_tslot = np.zeros((K_tslot*K_bw,1));
        self.Mean_bw_tslot[:] = math.inf;
        self.UCB_Index_bw_tslot = np.zeros((K_tslot*K_bw,1));  
        self.UCB_Index_bw_tslot[:] = math.inf;
        
        self.GMTS_Dir_par_bw_tslot_MCTS = np.ones((K_tslot,K_bw,M+1));  
        self.CountLeader_bw_tslot_MCTS = np.zeros((K_tslot,K_bw));
        self.NumUse_bw_tslot_MCTS = np.zeros((K_tslot,K_bw));
        self.Mean_bw_tslot_MCTS = np.zeros((K_tslot,K_bw));
        self.Mean_bw_tslot_MCTS[:] = math.inf;
        self.UCB_Index_bw_tslot_MCTS = np.zeros((K_tslot,K_bw));  
        self.UCB_Index_bw_tslot_MCTS[:] = math.inf;
        
        # Example: 4. Joint Beamwidth and beam direction selection (UCB+KLUCB+Correlation)       
        self.NumUse_beam = np.zeros((K_bw,K_beam_max));
        self.Mean_beam = np.zeros((K_bw,K_beam_max));
        self.Mean_beam[:] = -1;        
        self.UCB_Index_beam = np.zeros((K_bw,K_beam_max));
        for i in range(K_bw):
            self.UCB_Index_beam[i,:int(env_parameter.N_SSW_BS_vec[i])] = math.inf;
        self.new_reward = 100e9
        
        # Example: 5. Joint Beamwidth, t_slot and beam direction selection (UCB+KLUCB+Correlation)       
        self.NumUse_beam_3layers = np.zeros((K_tslot,K_bw,K_beam_max));
        self.Mean_beam_3layers = np.zeros((K_tslot,K_bw,K_beam_max));
        self.Mean_beam_3layers[:] = -1;        
        self.UCB_Index_beam_3layers = np.zeros((K_tslot,K_bw,K_beam_max));
        for j in range(K_tslot):
            for i in range(K_bw):
                self.UCB_Index_beam_3layers[j,i,:int(env_parameter.N_SSW_BS_vec[i])] = math.inf;
        
#-------------------------------------------------------------------------
# parameter for action
#-------------------------------------------------------------------------   
class def_action():
    def __init__(self, env_parameter):
        self.bw_id = 0;
        self.tslot_id = 1;
        self.beam_id = np.ones((max(env_parameter.N_SSW_BS_vec),1));