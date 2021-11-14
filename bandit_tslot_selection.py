# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb
from state_action import *
from calculate_UCB_Index import *

#-------------------------------------------------------------------------
# Bandit tslot selection
#------------------------------------------------------------------------- 
def bandit_tslot_selection_MTS(state, env_parameter):
    is_unimodal = True;
    K = env_parameter.K_tslot;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    UWMTS_Dir_par_tslot = state.UWMTS_Dir_par_tslot #np.ones((K_tslot,M+1));
    UCB_Index_tslot = state.UCB_Index_tslot #np.zeros((K_tslot,1));
    CountLeader_tslot = state.CountLeader_tslot #np.zeros((K_tslot,1));
    NumUse_tslot = state.NumUse_tslot #np.zeros((K_tslot,1));
    Mean_tslot = state.Mean_tslot #np.zeros((K_tslot,1));
    RateNor = env_parameter.RateNor;
    
    # Whether unimodality is exploited
    if is_unimodal:
        gamma_UWMTS = 3;
    else:
        gamma_UWMTS = 10e9;
    
    # Each arm is used at least once
    if np.isinf(np.amax(Mean_tslot)):
        posInf = np.where(Mean_tslot == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
        CountLeader_tslot[It] = 0;
    elif np.amin(NumUse_tslot) < env_parameter.min_use_per_tslot_guaranteed:
        It = np.argmin(NumUse_tslot)
    else:
        TSIndex = np.zeros(K);
        Leadert = np.argmax(Mean_tslot); # Decide leader at index t
        CountLeader_tslot[Leadert] += 1
        if (CountLeader_tslot[Leadert] % gamma_UWMTS) == 0:
            It = Leadert; # Decide the arm to play It
        else:
            if is_unimodal:
                neighbor = np.array([Leadert-1, Leadert, Leadert+1]);
            else:
                neighbor = np.array(range(0,K));
            for k in np.intersect1d(neighbor,np.array(range(0,K))):
                Lk = np.random.dirichlet(UWMTS_Dir_par_tslot[k,:]);                          
                TSIndex[k] = np.dot(RateNor,Lk);
            It = np.argmax(TSIndex); # Decide the arm to play It within the neighbors
            if len(np.intersect1d(It,neighbor)) == 0:
                sys.exit('Error in not choosing arm within neighbor of leaders in UWMTS algorithm');
    return It


#-------------------------------------------------------------------------
# Bandit parameter update for tslot selection
#------------------------------------------------------------------------- 
def update_bandit_tslot_para_MTS(env_parameter, state, action, reward, MCS_level):
    RateNor = env_parameter.RateNor;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    mt = MCS_level;
    UE_ID = 0;
    It = action.tslot_id;
    #-----Randomization-----
    Coeff_Eff = env_parameter.Coeff_BS2UE[action.bw_id,It] * (1-env_parameter.outage_coeff[UE_ID,UE_ID]);
    if np.random.binomial(1,Coeff_Eff) == 0:
        mt = 0; 
    #-----Randomization-----
    UWMTS_Dir_par_tslot = state.UWMTS_Dir_par_tslot;
    CountLeader_tslot = state.CountLeader_tslot;
    Mean_tslot = state.Mean_tslot;
    NumUse_tslot = state.NumUse_tslot;
    Leadert = np.argmax(Mean_tslot); # Decide leader at index t
    CountLeader_tslot[Leadert] += 1
    UWMTS_Dir_par_tslot[It,mt] = UWMTS_Dir_par_tslot[It,mt] + 1;
    #RateTmp = RateNor[mt]*Coeff_Eff;
    RateTmp = reward/env_parameter.maxRateValue;
    if np.isinf(Mean_tslot[It]):
        Mean_tslot[It] = RateTmp;
    else:
        Mean_tslot[It] = (Mean_tslot[It]*NumUse_tslot[It] + RateTmp)/(NumUse_tslot[It]+1);
    NumUse_tslot[It] += 1;
    state.UWMTS_Dir_par_tslot = UWMTS_Dir_par_tslot;
    state.CountLeader_tslot = CountLeader_tslot;
    state.Mean_tslot = Mean_tslot;
    state.NumUse_tslot = NumUse_tslot;
    return state


#-------------------------------------------------------------------------
# Bandit tslot selection
#------------------------------------------------------------------------- 
def bandit_tslot_selection_KL_UCB(state, env_parameter):    
    Mean_tslot = state.Mean_tslot
    NumUse_tslot = state.NumUse_tslot
    UCB_Index_tslot = state.UCB_Index_tslot        
    if np.isinf(np.amax(UCB_Index_tslot)): # Each arm is used at least once
        posInf = np.where(UCB_Index_tslot == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
    else:
        It = np.argmax(UCB_Index_tslot) # Decide the tslot to play It
    return It


#-------------------------------------------------------------------------
# Bandit parameter update for tslot selection
#------------------------------------------------------------------------- 
def update_bandit_tslot_para_KL_UCB(env_parameter, state, action, reward, MCS_level, ct):
    UCB_Constant = 4;
    Mean_tslot = state.Mean_tslot;
    NumUse_tslot = state.NumUse_tslot;
    UCB_Index_tslot = state.UCB_Index_tslot;
        
    # Reward: data rate
    RateTmp = reward/env_parameter.maxRateValue
    tslot_id = action.tslot_id
    bw_id = action.bw_id
            
    # Update first layer bandit parameters: Mean_tslot, NumUse_tslot and UCB_Index_tslot
    # Update mean
    if np.isinf(Mean_tslot[tslot_id]):
        Mean_tslot[tslot_id] = RateTmp;
    else:
        Mean_tslot[tslot_id] = (Mean_tslot[tslot_id]*NumUse_tslot[tslot_id] + RateTmp)/(NumUse_tslot[tslot_id]+1);
    # Update num use
    NumUse_tslot[tslot_id] = NumUse_tslot[tslot_id]+1;     
    # Update UCB index
    t = ct+1;
    KLUCB_Constant = 0;
    ft_KLUCB = 2*np.log(t) + KLUCB_Constant*np.log(np.log(t));
    ft_UCB = t**(2*UCB_Constant);
    UCB1_first_layer = False
    if UCB1_first_layer:
        UCB_Index_tslot[tslot_id] = Mean_tslot[tslot_id] + sqrt(log(ft_UCB)/NumUse_tslot[tslot_id]);
    else:
        epsilon = 0.00001;
        UCB_Index_tslot[tslot_id] = findKLUCB(Mean_tslot[tslot_id], ft_KLUCB, NumUse_tslot[tslot_id], epsilon);
    return state