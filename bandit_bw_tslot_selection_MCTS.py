# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb
import random
from state_action import *
from calculate_UCB_Index import *


#-------------------------------------------------------------------------
# Bandit beamwidth and beam direction selection
#------------------------------------------------------------------------- 
def bandit_bw_tslot_selection_MCTS(state, env_parameter, ct):
    # First-layer bandit for tslot selection
    Mean_tslot = state.Mean_tslot
    NumUse_tslot = state.NumUse_tslot
    UCB_Index_tslot = state.UCB_Index_tslot    
    if np.isinf(np.amax(UCB_Index_tslot)): # Each arm is used at least once
        posInf = np.where(UCB_Index_tslot == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
    elif np.amin(NumUse_tslot) < env_parameter.min_use_per_tslot_guaranteed:
        It = np.argmin(NumUse_tslot)
    else:
        It = np.argmax(UCB_Index_tslot) # Decide the tslot to play It
    tslot_id = It;
    
    # Second-layer bandit for beamwidth selection    
    Mean_bw_tslot_MCTS = state.Mean_bw_tslot_MCTS;
    NumUse_bw_tslot_MCTS = state.NumUse_bw_tslot_MCTS;
    UCB_Index_bw_tslot_MCTS = state.UCB_Index_bw_tslot_MCTS;
    if np.isinf(np.amax(UCB_Index_bw_tslot_MCTS[tslot_id,:])): # Each arm is used at least once
        posInf = np.where(UCB_Index_bw_tslot_MCTS[tslot_id,:] == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
    elif np.amin(NumUse_bw_tslot_MCTS[tslot_id,:]) < env_parameter.min_use_per_bw_guaranteed:
        It = np.argmin(NumUse_bw_tslot_MCTS[tslot_id,:])
    else:
        It = np.argmax(UCB_Index_bw_tslot_MCTS[tslot_id,:]) # Decide the bw to play It
    bw_id = It;
    
    return bw_id, tslot_id

#-------------------------------------------------------------------------
# Bandit parameter update for beamwidth and beam direction selection
#------------------------------------------------------------------------- 
def update_bandit_bw_tslot_para_MCTS(env_parameter, state, action, reward, MCS_level, ct):
    UCB_Constant = 4;
    Mean_tslot = state.Mean_tslot;
    NumUse_tslot = state.NumUse_tslot;
    UCB_Index_tslot = state.UCB_Index_tslot;
    Mean_bw_tslot_MCTS = state.Mean_bw_tslot_MCTS;
    NumUse_bw_tslot_MCTS = state.NumUse_bw_tslot_MCTS;
    UCB_Index_bw_tslot_MCTS = state.UCB_Index_bw_tslot_MCTS;
    N_SSW_BS_vec = env_parameter.N_SSW_BS_vec;
    ratio_codebook_used = env_parameter.ratio_codebook_used;
    sub_N_SSW_BS_vec = N_SSW_BS_vec * ratio_codebook_used;
        
    # Reward: data rate
    RateTmp = reward/env_parameter.maxRateValue
    new_reward = reward;
    tslot_id = action.tslot_id
    bw_id = action.bw_id
    
    # Update second lyaer bandit parameters: Mean_tslot_bw_MCTS, NumUse_tslot_bw_MCTS and UCB_Index_tslot_bw_MCTS    
    new_t = NumUse_tslot[tslot_id]+1+0.0001;
    ft_UCB = new_t**(2*UCB_Constant);
    ft_KLUCB = 2*np.log(new_t) + 3*np.log(np.log(new_t));
    # Update mean
    if np.isinf(Mean_bw_tslot_MCTS[tslot_id,bw_id]):
        Mean_bw_tslot_MCTS[tslot_id,bw_id] = RateTmp;
    else:
        Mean_bw_tslot_MCTS[tslot_id,bw_id] = (Mean_bw_tslot_MCTS[tslot_id,bw_id]*NumUse_bw_tslot_MCTS[tslot_id,bw_id] + RateTmp)/(NumUse_bw_tslot_MCTS[tslot_id,bw_id]+1);
    # Update num use
    NumUse_bw_tslot_MCTS[tslot_id,bw_id] = NumUse_bw_tslot_MCTS[tslot_id,bw_id]+1;     
    # Update UCB index
    UCB1_first_layer = False
    if UCB1_first_layer:
        UCB_Index_bw_tslot_MCTS[tslot_id,bw_id] = Mean_bw_tslot_MCTS[tslot_id,bw_id] + sqrt(log(ft_UCB)/NumUse_bw_tslot_MCTS[tslot_id,bw_id]);
    else:
        epsilon = 0.00001;
        UCB_Index_bw_tslot_MCTS[tslot_id,bw_id] = findKLUCB(Mean_bw_tslot_MCTS[tslot_id,bw_id], ft_KLUCB, NumUse_bw_tslot_MCTS[tslot_id,bw_id], epsilon);
        
    # Update first layer bandit parameters: Mean_tslot, NumUse_tslot and UCB_Index_tslot
    t = ct+1+0.0001;
    ft_KLUCB = 2*np.log(t) + 3*np.log(np.log(t));
    ft_UCB = t**(2*UCB_Constant);
    # Update mean
    if np.isinf(Mean_tslot[tslot_id]):
        Mean_tslot[tslot_id] = RateTmp;
    else:
        Mean_tslot[tslot_id] = (Mean_tslot[tslot_id]*NumUse_tslot[tslot_id] + RateTmp)/(NumUse_tslot[tslot_id]+1);
    # Update num use
    NumUse_tslot[tslot_id] = NumUse_tslot[tslot_id]+1;     
    # Update UCB index
    if np.any(UCB_Index_bw_tslot_MCTS[tslot_id,:] == np.inf):
        UCB_Index_tslot[tslot_id] = np.inf;
    else:
        UCB1_first_layer = False
        if UCB1_first_layer:
            UCB_Index_tslot[tslot_id] = Mean_tslot[tslot_id] + sqrt(log(ft_UCB)/NumUse_tslot[tslot_id]);
        else:
            epsilon = 0.00001;
            UCB_Index_tslot[tslot_id] = findKLUCB(Mean_tslot[tslot_id], ft_KLUCB, NumUse_tslot[tslot_id], epsilon);
        
    state.Mean_tslot = Mean_tslot;
    state.NumUse_tslot = NumUse_tslot;
    state.UCB_Index_tslot = UCB_Index_tslot;
    state.Mean_bw_tslot_MCTS = Mean_bw_tslot_MCTS;
    state.NumUse_bw_tslot_MCTS = NumUse_bw_tslot_MCTS;
    state.UCB_Index_bw_tslot_MCTS = UCB_Index_bw_tslot_MCTS;    
    state.new_reward = new_reward;
    
    #pdb.set_trace()
    
    return state


