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
# import warnings
# warnings.filterwarnings("error")


#-------------------------------------------------------------------------
# Bandit beamwidth and beam direction selection
#------------------------------------------------------------------------- 
def bandit_tslot_bw_beam_3layers_selection_MCTS(state, env_parameter, ct):
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
    
    # Third-layer bandit for Top-1/4 beam selection    
    Mean_beam_3layers = state.Mean_beam_3layers;
    NumUse_beam_3layers = state.NumUse_beam_3layers;
    UCB_Index_beam_3layers = state.UCB_Index_beam_3layers;
    N_SSW_BS_vec = env_parameter.N_SSW_BS_vec;
    N_SSW_BS_vec = N_SSW_BS_vec.astype(int);
    ratio_codebook_used = env_parameter.ratio_codebook_used;
    sub_N_SSW_BS_vec = N_SSW_BS_vec * ratio_codebook_used; 
    posInf = np.where(UCB_Index_beam_3layers[tslot_id,bw_id,:N_SSW_BS_vec[bw_id]] == np.inf)[0]
    if len(posInf) > sub_N_SSW_BS_vec[bw_id]: # Each arm is used at least once
        sorted_ind_selected = posInf[random.sample(range(len(posInf)), int(sub_N_SSW_BS_vec[bw_id]))]
    else:
        sorted_ind = np.argsort(-UCB_Index_beam_3layers[tslot_id,bw_id,:N_SSW_BS_vec[bw_id]]);
        sorted_ind_selected = sorted_ind[:int(sub_N_SSW_BS_vec[bw_id])]
    beam_id = np.zeros((max(N_SSW_BS_vec),1));
    beam_id[sorted_ind_selected] = 1;   
    
    return tslot_id, bw_id, beam_id


#-------------------------------------------------------------------------
# Bandit parameter update for beamwidth and beam direction selection
#------------------------------------------------------------------------- 
def update_bandit_tslot_bw_beam_3layers_para_MCTS(env_parameter, state, action, reward, MCS_level, enable_correlation, ct):
    UCB_Constant = 4
    KLUCB_Constant = 0;
    Mean_beam_3layers = state.Mean_beam_3layers;
    NumUse_beam_3layers = state.NumUse_beam_3layers;
    UCB_Index_beam_3layers = state.UCB_Index_beam_3layers;    
    
    Mean_tslot = state.Mean_tslot;
    NumUse_tslot = state.NumUse_tslot;
    UCB_Index_tslot = state.UCB_Index_tslot;
    
    Mean_bw_tslot_MCTS = state.Mean_bw_tslot_MCTS;
    NumUse_bw_tslot_MCTS = state.NumUse_bw_tslot_MCTS;
    UCB_Index_bw_tslot_MCTS = state.UCB_Index_bw_tslot_MCTS;
    
    N_SSW_BS_vec = env_parameter.N_SSW_BS_vec;
    ratio_codebook_used = env_parameter.ratio_codebook_used;
    sub_N_SSW_BS_vec = N_SSW_BS_vec * ratio_codebook_used;
        
    # Reward of second layer: realization of connectivity (using position to calculate)
    i_sel = action.bw_id
    j_sel = np.where(action.beam_id == 1)[0]
    j_sel = j_sel.astype(int)
    I = env_parameter.K_bw
    r = np.ones((I,1))*N_SSW_BS_vec[1]/N_SSW_BS_vec[0];
    r = r.astype(int)
    M = N_SSW_BS_vec.astype(int)
    K = sub_N_SSW_BS_vec.astype(int)
    success = np.zeros((I,M[I-1]));
    
    # Using position to calculate the realization of successful connection
    Xcoor_UE = env_parameter.Xcoor_list[0][-2]
    Ycoor_UE = env_parameter.Xcoor_list[0][-2]
    Xcoor_AP = env_parameter.Xcoor_list[-1][-2]
    Ycoor_AP = env_parameter.Xcoor_list[-1][-2]
    dirc = np.arctan2(Ycoor_UE-Ycoor_AP,Xcoor_UE-Xcoor_AP)
    if dirc<0:
        dirc = dirc + 2*np.pi
    highest_resolution_beam_3layers = 2*np.pi/M[-1];
    active_beam_3layers_id = int(dirc // highest_resolution_beam_3layers)
    success[I-1,active_beam_3layers_id] = 1;
    # Propagate the ground truth of the status of success
    for i in range(I-2,-1,-1): 
        for j in range(M[i]):
            success[i,j] = min(np.sum(success[i+1,int(j*r[i+1]):int((j+1)*r[i+1])]),1);            
            
    # Reward of first layer: data rate
    RateTmp = reward/env_parameter.maxRateValue * np.sum(success[i_sel,j_sel])
    if RateTmp == 0:
        new_reward = 0;
    else:
        new_reward = reward;
    
    #
    tslot_id = action.tslot_id
    bw_id = action.bw_id
    
    # Update third layer bandit parameters: Mean_beam, NumUse_beam and UCB_Index_beam 
    Mean_beam_3layers_tmp, NumUse_beam_3layers_tmp, UCB_Index_beam_3layers_tmp = \
    calculate_UCB_Index(enable_correlation,success,\
                        np.squeeze(Mean_beam_3layers[tslot_id,:,:]),\
                        np.squeeze(NumUse_beam_3layers[tslot_id,:,:]),\
                        np.squeeze(UCB_Index_beam_3layers[tslot_id,:,:]),\
                        i_sel,j_sel,I,r,M,K,NumUse_bw_tslot_MCTS[tslot_id,bw_id]);
    Mean_beam_3layers[tslot_id,:,:] = Mean_beam_3layers_tmp;
    NumUse_beam_3layers[tslot_id,:,:] = NumUse_beam_3layers_tmp;
    UCB_Index_beam_3layers[tslot_id,:,:] = UCB_Index_beam_3layers_tmp;
    
    # Update second lyaer bandit parameters: Mean_tslot_bw_MCTS, NumUse_tslot_bw_MCTS and UCB_Index_tslot_bw_MCTS    
    new_t = NumUse_tslot[tslot_id]+1+0.0001;
    ft_KLUCB = 2*np.log(new_t) + KLUCB_Constant*np.log(np.log(new_t));
    ft_UCB = new_t**(2*UCB_Constant);
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
    ft_KLUCB = 2*np.log(t) + KLUCB_Constant*np.log(np.log(t));
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
    
    state.Mean_beam_3layers = Mean_beam_3layers;
    state.NumUse_beam_3layers = NumUse_beam_3layers;
    state.UCB_Index_beam_3layers = UCB_Index_beam_3layers;    
    state.new_reward = new_reward;
    
    #pdb.set_trace()
    
    return state


