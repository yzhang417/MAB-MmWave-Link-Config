# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb
from state_action import *

#-------------------------------------------------------------------------
# Bandit bw_tslot selection
#------------------------------------------------------------------------- 
def bandit_bw_tslot_selection_MTS(state, env_parameter):
    K = env_parameter.K_bw * env_parameter.K_tslot;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    GMTS_Dir_par_bw_tslot = state.GMTS_Dir_par_bw_tslot #np.ones((K_bw*K_tslot,M+1));
    UCB_Index_bw_tslot = state.UCB_Index_bw_tslot #np.zeros((K_bw*K_tslot,M+1));
    NumUse_bw_tslot = state.NumUse_bw_tslot #np.zeros((K_bw*K_tslot,M+1));
    Mean_bw_tslot = state.Mean_bw_tslot #np.zeros((K_bw*K_tslot,M+1));
    RateNor = env_parameter.RateNor;    
    # Each arm is used at least once
    if np.isinf(np.amax(Mean_bw_tslot)):
        posInf = np.where(Mean_bw_tslot == np.inf)[0]
        It = posInf[np.random.randint(len(posInf))]
    elif np.amin(NumUse_bw_tslot) < env_parameter.min_use_per_bw_tslot_guaranteed:
        It = np.argmin(NumUse_bw_tslot)
    else:
        TSIndex = np.zeros(K);
        for k in range(K):
            Lk = np.random.dirichlet(GMTS_Dir_par_bw_tslot[k,:]);                          
            TSIndex[k] = np.dot(RateNor,Lk);
        It = np.argmax(TSIndex); # Decide the arm to play It
    bw_id = It // env_parameter.K_tslot
    tslot_id = It % env_parameter.K_tslot
    return bw_id, tslot_id


#-------------------------------------------------------------------------
# Bandit parameter update for bw_tslot selection
#------------------------------------------------------------------------- 
def update_bandit_bw_tslot_para_MTS(env_parameter, state, action, reward, MCS_level):
    RateNor = env_parameter.RateNor;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    mt = MCS_level;
    UE_ID = 0;
    It = action.bw_id * env_parameter.K_tslot + action.tslot_id;
    #-----Randomization-----
    Coeff_Eff = env_parameter.Coeff_BS2UE[action.bw_id,action.tslot_id] * (1-env_parameter.outage_coeff[UE_ID,UE_ID]);
    if np.random.binomial(1,Coeff_Eff) == 0:
        mt = 0; 
    #-----Randomization-----
    GMTS_Dir_par_bw_tslot = state.GMTS_Dir_par_bw_tslot;
    Mean_bw_tslot = state.Mean_bw_tslot;
    NumUse_bw_tslot = state.NumUse_bw_tslot;
    GMTS_Dir_par_bw_tslot[It,mt] = GMTS_Dir_par_bw_tslot[It,mt] + 1;
    RateTmp = RateNor[mt]*Coeff_Eff;
    if np.isinf(Mean_bw_tslot[It]):
        Mean_bw_tslot[It] = RateTmp;
    else:
        Mean_bw_tslot[It] = (Mean_bw_tslot[It]*NumUse_bw_tslot[It] + RateTmp)/(NumUse_bw_tslot[It]+1);
    NumUse_bw_tslot[It] += 1;
    state.GMTS_Dir_par_bw_tslot = GMTS_Dir_par_bw_tslot;
    state.Mean_bw_tslot = Mean_bw_tslot;
    state.NumUse_bw_tslot = NumUse_bw_tslot;
    return state