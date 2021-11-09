# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb
from state_action import *
from bandit_bw_selection import *
from bandit_tslot_selection import *
from bandit_bw_tslot_selection import *
from bandit_bw_tslot_selection_MCTS import *
from bandit_bw_beam_selection import *
from bandit_tslot_bw_beam_selection_MCTS import *


#-------------------------------------------------------------------------
# Decision making function BW selection
#------------------------------------------------------------------------- 
def decision_making(state, env_parameter, exid, policy_setting, ct):
    action = def_action(env_parameter)
    # Beamwidth selection (K_bw + 2 polices)
    if exid == 1:
        action.tslot_id = 1;
        action.beam_id = np.ones((max(env_parameter.N_SSW_BS_vec),1));
        if policy_setting.enable_bandit is True:
            action.bw_id = bandit_bw_selection_MTS(state, env_parameter);
        elif policy_setting.enable_random is True:
            action.bw_id = np.random.randint(env_parameter.K_bw);
        else:
            action.bw_id = policy_setting.default_bw_id
    # Beam training periodicity selection (K_tslot + 2 polices)
    elif exid == 2:
        action.bw_id = 4;
        action.beam_id = np.ones((max(env_parameter.N_SSW_BS_vec),1));
        if policy_setting.enable_bandit is True:
            if policy_setting.enable_TS:
                action.tslot_id = bandit_tslot_selection_MTS(state, env_parameter);
            else:
                action.tslot_id = bandit_tslot_selection_KL_UCB(state, env_parameter);
        elif policy_setting.enable_random is True:
            action.tslot_id = np.random.randint(env_parameter.K_tslot);
        else:
            action.tslot_id = policy_setting.default_tslot_id
    # Joint Beamwidth and Beam training periodicity selection (K_tslot * K_bw + 2 polices)
    elif exid == 3:
        action.beam_id = np.ones((max(env_parameter.N_SSW_BS_vec),1));
        if policy_setting.enable_monte_carlo is True:
            action.bw_id, action.tslot_id = bandit_bw_tslot_selection_MCTS(state, env_parameter, ct);
        elif policy_setting.enable_bandit is True:
            action.bw_id, action.tslot_id = bandit_bw_tslot_selection_MTS(state, env_parameter);
        elif policy_setting.enable_random is True:
            action.tslot_id = np.random.randint(env_parameter.K_tslot);
            action.bw_id = np.random.randint(env_parameter.K_bw);
        else:
            action.tslot_id = policy_setting.default_tslot_id
            action.bw_id = policy_setting.default_bw_id
    # Joint Beamwidth and beam direction selection (3 policies)
    elif exid == 4:
        action.tslot_id = 1;
        if policy_setting.enable_bandit is True:
            action.bw_id, action.beam_id = bandit_bw_beam_selection_UCB(state, env_parameter, ct);            
        elif policy_setting.enable_random is True:
            action.bw_id = np.random.randint(env_parameter.K_bw);
            action.beam_id = np.ones((max(env_parameter.N_SSW_BS_vec),1));
        else:
            action.bw_id = policy_setting.default_bw_id
            action.beam_id = policy_setting.default_beam_id
    else:
        action.tslot_id = 1;
        if policy_setting.enable_bandit is True:
            action.tslot_id, action.bw_id, action.beam_id = bandit_tslot_bw_beam_3layers_selection_MCTS(state, env_parameter, ct)
        elif policy_setting.enable_random is True:
            action.tslot_id = np.random.randint(env_parameter.K_tslot);
            action.bw_id = np.random.randint(env_parameter.K_bw);
            action.beam_id = np.ones((max(env_parameter.N_SSW_BS_vec),1));
        else:
            action.tslot_id = npolicy_setting.default_tslot_id
            action.bw_id = policy_setting.default_bw_id
            action.beam_id = policy_setting.default_beam_id
    return action


#-------------------------------------------------------------------------
# Decision making function BW selection
#------------------------------------------------------------------------- 
def update_bandit_para(env_parameter, state, action, reward, MCS_level, exid, policy_setting, ct):
    # Beamwidth selection (K_bw + 2 polices)
    if exid == 1:
        if policy_setting.enable_bandit is True:
            return update_bandit_bw_para_MTS(env_parameter, state, action, reward, MCS_level)
        else:
            return state
    # Beam training periodicity selection (K_tslot + 2 polices)
    elif exid == 2:
        if policy_setting.enable_bandit is True:
            if policy_setting.enable_TS:
                return update_bandit_tslot_para_MTS(env_parameter, state, action, reward, MCS_level)
            else:
                return update_bandit_tslot_para_KL_UCB(env_parameter, state, action, reward, MCS_level, ct)
        else:
            return state
    # Joint Beamwidth and Beam training periodicity selection (K_tslot * K_bw + 2 polices)
    elif exid == 3:
        if policy_setting.enable_monte_carlo is True:
            return update_bandit_bw_tslot_para_MCTS(env_parameter, state, action, reward, MCS_level, ct)
        elif policy_setting.enable_bandit is True:
            return update_bandit_bw_tslot_para_MTS(env_parameter, state, action, reward, MCS_level)
        else:
            return state
    # Joint Beamwidth and beam direction selection (3 policies)
    elif exid == 4:
        if policy_setting.enable_bandit is True:
            enable_correlation = policy_setting.enable_correlation
            return update_bandit_bw_beam_para_UCB(env_parameter, state, action, reward, MCS_level, enable_correlation, ct)
        else:
            return state
    else:
        if policy_setting.enable_bandit is True:
            enable_correlation = policy_setting.enable_correlation
            return update_bandit_tslot_bw_beam_3layers_para_MCTS(env_parameter, state, action, reward, MCS_level, enable_correlation, ct)
        else:
            return state
        
        
        