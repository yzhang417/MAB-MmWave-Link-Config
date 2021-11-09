# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb
from state_action import *

#-------------------------------------------------------------------------
# Bandit BW selection
#------------------------------------------------------------------------- 
def bandit_bw_selection_MTS(state, env_parameter):
    is_unimodal = True;
    K = env_parameter.K_bw;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    UWMTS_Dir_par_bw = state.UWMTS_Dir_par_bw #np.ones((K_bw,M+1));
    UCB_Index_bw = state.UCB_Index_bw #np.zeros((K_bw,1));
    CountLeader_bw = state.CountLeader_bw #np.zeros((K_bw,1));
    NumUse_bw = state.NumUse_bw #np.zeros((K_bw,1));
    Mean_bw = state.Mean_bw #np.zeros((K_bw,1));
    RateNor = env_parameter.RateNor;
    
    # Whether unimodality is exploited
    if is_unimodal:
        gamma_UWMTS = 3;
    else:
        gamma_UWMTS = 10e9;
    
    # Each arm is used at least once
    if np.isinf(np.amax(Mean_bw)):
        posInf = np.where(Mean_bw == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
        CountLeader_bw[It] = 0;
    elif np.amin(NumUse_bw) < env_parameter.min_use_per_bw_guaranteed:
        It = np.argmin(NumUse_bw)
    else:
        TSIndex = np.zeros(K);
        Leadert = np.argmax(Mean_bw); # Decide leader at index t
        CountLeader_bw[Leadert] += 1
        if (CountLeader_bw[Leadert] % gamma_UWMTS) == 0:
            It = Leadert; # Decide the arm to play It
        else:
            if is_unimodal:
                neighbor = np.array([Leadert-1, Leadert, Leadert+1]);
            else:
                neighbor = np.array(range(0,K));
            for k in np.intersect1d(neighbor,np.array(range(0,K))):
                Lk = np.random.dirichlet(UWMTS_Dir_par_bw[k,:]);                          
                TSIndex[k] = np.dot(RateNor,Lk);
            It = np.argmax(TSIndex); # Decide the arm to play It within the neighbors
            if len(np.intersect1d(It,neighbor)) == 0:
                sys.exit('Error in not choosing arm within neighbor of leaders in UWMTS algorithm');
    return It

def bandit_bw_selection_KLUCB(state, env_parameter):

    return It


#-------------------------------------------------------------------------
# Bandit parameter update for bw selection
#------------------------------------------------------------------------- 
def update_bandit_bw_para_MTS(env_parameter, state, action, reward, MCS_level):
    RateNor = env_parameter.RateNor;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    mt = MCS_level;
    UE_ID = 0;
    It = action.bw_id;
    #-----Randomization-----
    Coeff_Eff = env_parameter.Coeff_BS2UE[It,action.tslot_id] * (1-env_parameter.outage_coeff[UE_ID,UE_ID]);
    if np.random.binomial(1,Coeff_Eff) == 0:
        mt = 0; 
    #-----Randomization-----
    UWMTS_Dir_par_bw = state.UWMTS_Dir_par_bw;
    CountLeader_bw = state.CountLeader_bw;
    Mean_bw = state.Mean_bw;
    NumUse_bw = state.NumUse_bw;
    Leadert = np.argmax(Mean_bw); # Decide leader at index t
    CountLeader_bw[Leadert] += 1
    UWMTS_Dir_par_bw[It,mt] = UWMTS_Dir_par_bw[It,mt] + 1;
    RateTmp = RateNor[mt]*Coeff_Eff;
    if np.isinf(Mean_bw[It]):
        Mean_bw[It] = RateTmp;
    else:
        Mean_bw[It] = (Mean_bw[It]*NumUse_bw[It] + RateTmp)/(NumUse_bw[It]+1);
    NumUse_bw[It] += 1;
    state.UWMTS_Dir_par_bw = UWMTS_Dir_par_bw;
    state.CountLeader_bw = CountLeader_bw;
    state.Mean_bw = Mean_bw;
    state.NumUse_bw = NumUse_bw;
    return state