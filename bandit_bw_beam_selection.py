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
def bandit_bw_beam_selection_UCB(state, env_parameter, ct):
    # First-layer bandit for beamwidth selection
    Mean_bw = state.Mean_bw
    NumUse_bw = state.NumUse_bw
    UCB_Index_bw = state.UCB_Index_bw    
    if np.isinf(np.amax(UCB_Index_bw)): # Each arm is used at least once
        posInf = np.where(UCB_Index_bw == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
    elif np.amin(NumUse_bw) < env_parameter.min_use_per_bw_guaranteed:
        It = np.argmin(NumUse_bw)
    else:
        It = np.argmax(UCB_Index_bw) # Decide the bw to play It
    bw_id = It    
    
    # Second-layer bandit for Top-1/4 beam selection    
    Mean_beam = state.Mean_beam;
    NumUse_beam = state.NumUse_beam;
    UCB_Index_beam = state.UCB_Index_beam;
    N_SSW_BS_vec = env_parameter.N_SSW_BS_vec;
    N_SSW_BS_vec = N_SSW_BS_vec.astype(int);
    ratio_codebook_used = env_parameter.ratio_codebook_used;
    sub_N_SSW_BS_vec = N_SSW_BS_vec * ratio_codebook_used; 
    posInf = np.where(UCB_Index_beam[bw_id,:N_SSW_BS_vec[bw_id]] == np.inf)[0]
    if len(posInf) > sub_N_SSW_BS_vec[bw_id]: # Each arm is used at least once
        sorted_ind_selected = posInf[random.sample(range(len(posInf)), int(sub_N_SSW_BS_vec[bw_id]))]
    else:
        sorted_ind = np.argsort(-UCB_Index_beam[bw_id,:N_SSW_BS_vec[bw_id]]);
        sorted_ind_selected = sorted_ind[:int(sub_N_SSW_BS_vec[bw_id])]
    beam_id = np.zeros((max(N_SSW_BS_vec),1));
    beam_id[sorted_ind_selected] = 1;   
    
    return bw_id, beam_id


#-------------------------------------------------------------------------
# Bandit parameter update for beamwidth and beam direction selection
#------------------------------------------------------------------------- 
def update_bandit_bw_beam_para_UCB(env_parameter, state, action, reward, MCS_level, enable_correlation, ct):
    t = ct+1;
    ft_KLUCB = 2*np.log(t) + 3*np.log(np.log(t));
    UCB_Constant = 4
    ft_UCB = t^(2*UCB_Constant);
    Mean_bw = state.Mean_bw;
    NumUse_bw = state.NumUse_bw;
    UCB_Index_bw = state.UCB_Index_bw;
    Mean_beam = state.Mean_beam;
    NumUse_beam = state.NumUse_beam;
    UCB_Index_beam = state.UCB_Index_beam;
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
    highest_resolution_beam = 2*np.pi/M[-1];
    active_beam_id = int(dirc // highest_resolution_beam)
    success[I-1,active_beam_id] = 1;
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
    
    # Update first layer bandit parameters: Mean_bw, NumUse_bw and UCB_Index_bw
    bw_id = action.bw_id
    # Update mean
    if np.isinf(Mean_bw[bw_id]):
        Mean_bw[bw_id] = RateTmp;
    else:
        Mean_bw[bw_id] = (Mean_bw[bw_id]*NumUse_bw[bw_id] + RateTmp)/(NumUse_bw[bw_id]+1);
    # Update num use
    NumUse_bw[bw_id] = NumUse_bw[bw_id]+1;     
    
    # Update UCB index of bw
    if np.any(UCB_Index_beam[bw_id,:N_SSW_BS_vec[bw_id]] == np.inf):
        UCB_Index_bw[bw_id] = np.inf;
    else:
        UCB1_first_layer = False
        if UCB1_first_layer:
            UCB_Index_bw[bw_id] = Mean_bw[bw_id] + sqrt(log(ft_UCB)/NumUse_bw[bw_id]);
        else:
            epsilon = 0.00001;
            UCB_Index_bw[bw_id] = findKLUCB(Mean_bw[bw_id], ft_KLUCB, NumUse_bw[bw_id], epsilon);
    state.Mean_bw = Mean_bw;
    state.NumUse_bw = NumUse_bw;
    state.UCB_Index_bw = UCB_Index_bw;
    
    # Update second layer bandit parameters: Mean_bw, NumUse_bw and UCB_Index_bw 
    Mean_beam, NumUse_beam, UCB_Index_beam = calculate_UCB_Index(enable_correlation,success,Mean_beam,NumUse_beam,UCB_Index_beam,i_sel,j_sel,I,r,M,K,NumUse_bw[bw_id])
    state.Mean_beam = Mean_beam;
    state.NumUse_beam = NumUse_beam;
    state.UCB_Index_beam = UCB_Index_beam;    
    state.new_reward = new_reward;
    
    #pdb.set_trace()
    
    return state


