#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import operator
import os
import math
import random
import pdb
from state_action import *
from my_utils import *
import copy
# from scipy.ndimage.interpolation import shift
# numpy.pad is much more efficient than shift


#-----------------------------------------------------------------------
# Class of parameters configuring the wireless system
#-----------------------------------------------------------------------
class def_env_para():
    def __init__(self):
        pass
    

#-----------------------------------------------------------------------
# Following function allows configuring the studied wirelss system and 
# will return an instance of def_env_para consisting of necesary parameters
#-----------------------------------------------------------------------
def env_init(ratio_codebook_used=1):    
    
    # -------------------------------------
    # Network topology
    # -------------------------------------    
    N_UE = 5; # Number of users 
    radius = np.array([20, 10, 20, 30, 40]);  # Distance between Tx and Rx
    angle =  np.array([45, 0, 0, 0, 0]);    # Angle between Tx and Rx
    target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.05, 0.05, 0.05]); # Average percentage of slots in blockage
    target_prob_blockage_D2D = 0.05
    target_prob_blockage = np.ones((N_UE,N_UE)) - np.diag(np.ones(N_UE))
    target_prob_blockage = target_prob_blockage * target_prob_blockage_D2D + np.diag(target_prob_blockage_to_AP)
    Xcoor_init, Ycoor_init = pol2cart(np.deg2rad(angle),radius);
    Xcoor_init = np.append(Xcoor_init,0) # Last index for AP X
    Ycoor_init = np.append(Ycoor_init,0) # Last index for AP Y

    
    # -----------------------------------------------
    # Beam training setting (realignment periodicity)
    # -----------------------------------------------
    t_slot_vec = np.array([10,20,40,80,160])*1e-3;       # Time duration for a single slot in seconds
    t_SSW = 10 * 1e-6;        # Time of SSW frame in seconds (time duration for per measurement)
    Num_arr_AP = 4;           # Number of antenna arrays equipped to cover 360 degrees
    Num_arr_UE = 4;           # Number of antenna arrays equipped to cover 360 degrees
    Beampair_Repetition = 1;  # Repetion of each beam pair sounding
    BeamWidth_vertical = 75;  # Elevation beamwidth
    single_side_beam_training = False;    # By default, we consider double side beam training    
    
    
    # -------------------------------------
    # Mobility model
    # -------------------------------------
    max_activity_range = 10; # maximum distance in meters from the initial position
    v_min = [5,0,0,0,0,0];  # minimum speed in m/s
    v_max = [15,0,0,0,0,0]; # maximum speed in m/s
    last_direction = np.ones(N_UE+1)*np.deg2rad(angle[0]-90) # last index is for the mobility of AP
    last_velocity = np.ones(N_UE+1)*(v_min[0]+v_max[0])/2 # last index is for the mobility of AP
    v_self_rotate_min = 0  # minimum ratation speed in degrees/s
    v_self_rotate_max = 10  # maximum ratation speed in degrees/s
    max_number_last_rotate = 10
    max_number_last_direction = 10
    number_last_rotate = np.ones(N_UE+1) * max_number_last_rotate # last index is for the mobility of AP
    number_last_direction = np.ones(N_UE+1) * max_number_last_direction  # last index is for the mobility of AP

    
    
    # -------------------------------------
    # Blockage model
    # -------------------------------------
    blockage_loss_dB = 20;     # Blockage loss
    min_blockage_duration = 2; # Number of minimum slots that blockage exists
    max_blockage_duration = 6; # Number of maximum slots that blockage exists
    min_blockage_duration_guess = 1;   # Guess of min_blockage_duration
    max_blockage_duration_guess = 10;  # Guess of max_blockage_duration
    prob_blockage = target_prob_blockage/\
    (target_prob_blockage+(min_blockage_duration+max_blockage_duration)/2*(1-target_prob_blockage))  

    
    # -------------------------------------
    # Channel correlation and fading model
    # -------------------------------------
    #channel_corr_coeff = np.array([0.8,0.2]); # Channel time correlation
    channel_corr_coeff = np.array([1]);
    mean_X_coeff = 0;
    sigma_X_coeff = np.sqrt(4);
    mean_X = mean_X_coeff*np.zeros((N_UE,N_UE));
    sigma_X = sigma_X_coeff*np.ones((N_UE,N_UE));

    
    # -------------------------------------
    # PHY/MAC parameters
    # -------------------------------------
    W1 = 5;             # Link marginal budget
    W2 = 5;             # Implementation Loss
    W = W1 + W2;        # Total link budget
    B = 2160 * 1e6;     # Bandwith in Hz
    Using_MCS = 1;      # Using MCS and sensibility of IEEE 802.11 ad standard (Can be updated to 5G NR)
    MCS = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.1, 10, 11, 12, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6]);
    RS_Value = np.array([-1000, -78, -68, -66, -65, -64, -63, -62, -61, -60, -59, -57, -55, -54, -53, -51, -50, -48, -46, -44, -42]);
    Rate_Value = np.array([0, 27.5, 385, 770, 962.5, 1155, 1251.25, 1540, 1925, 2310, 2502.5, 2695, 3080, 3850, 4620, 5005, 5390, 5775, 6390, 7507.5, 8085])*1e6;
    maxRateValue = np.amax(Rate_Value);    # Maximum rate supported (Highest MCS)
    RateNor = Rate_Value/maxRateValue;     # Normalized rate within [0,1] so that the reward is bounded
    M = len(Rate_Value)-1;                 # Number of MCS, aka size of supports without 0 bps rate option
    Pn_dBm = -174 + 10*np.log10(B) + 10;   # Power of noise in dBm
    SNR_Value = RS_Value - Pn_dBm;         # Corresponding SNR table to RSS


    # -------------------------------------
    # Antenna setting at BS and UE
    # -------------------------------------
    N_SSW_BS_vec = np.array([16, 32, 64, 128, 256, 512]); # Tx total number of sectors to cover 2D space
    BeamWidth_TX_vec = 360./N_SSW_BS_vec;   # Tx antenna beamwidth
    N_SSW_UE_vec = 16;                      # Rx total number of sectors to cover 2D space
    BeamWidth_RX_vec = 360/N_SSW_UE_vec;    # Rx antenna beamwidth
    fc = 60;                                # carrier frequency in GHz
    c = 3e8;                                # speed of light
    l = c/(fc*1e9);                         # wavelength
    d = l/2;                                # antenna spacing
    
    
    # -------------------------------------
    # Antenna gain between BS and UEs
    # -------------------------------------
    Ptx_BS_dBm = 15;   # Power of transmitting signal from BS in dBm
    Gt_vec = 16*np.pi/(6.67*np.deg2rad(BeamWidth_TX_vec)*np.deg2rad(BeamWidth_vertical)); # Transmitting antenna gain
    Gt_dBi_vec = 10*np.log10(Gt_vec);                      # Transmitting antenna gain in dBi
    Gr_vec = 16*np.pi/(6.67*np.deg2rad(BeamWidth_RX_vec)*np.deg2rad(BeamWidth_vertical)); # Receiving antenna gain
    Gr_dBi_vec = 10*np.log10(Gr_vec);                      # Receiving antenna gain in dBi
    EIRP = 43;                                             # Limitation of EIRP in USA by FCC
    EIRP_real_max = np.amax(Gt_dBi_vec) + Ptx_BS_dBm;
    if EIRP_real_max > EIRP:
        print('Error in EIRP for BS');     # Validate that the EIRP meets the FCC requirement
    GtGr_dBi_mat = Gt_dBi_vec + Gr_dBi_vec;

    
    # -------------------------------------
    # Saved enviroment parameter
    # -------------------------------------
    env_parameter = def_env_para(); 
    env_parameter.N_UE = N_UE;
    env_parameter.Rate_Value = Rate_Value;
    env_parameter.maxRateValue = maxRateValue;
    env_parameter.RateNor = RateNor;
    env_parameter.M = M;
    env_parameter.SNR_Value = SNR_Value;
    env_parameter.N_SSW_BS_vec = N_SSW_BS_vec;
    env_parameter.N_SSW_UE_vec = N_SSW_UE_vec;
    env_parameter.W = W;
    env_parameter.B = B;
    env_parameter.fc = fc;
    env_parameter.t_slot_vec = t_slot_vec;
    env_parameter.t_SSW = t_SSW;
    env_parameter.BeamWidth_TX_vec = BeamWidth_TX_vec;
    env_parameter.BeamWidth_RX_vec = BeamWidth_RX_vec;
    env_parameter.Ptx_BS_dBm = Ptx_BS_dBm;    
    env_parameter.GtGr_dBi_mat = GtGr_dBi_mat;
    env_parameter.Pn_dBm = Pn_dBm;
    env_parameter.Using_MCS = Using_MCS;
    env_parameter.Num_arr_AP = Num_arr_AP;    
    env_parameter.Num_arr_UE = Num_arr_UE;           
    env_parameter.Beampair_Repetition = Beampair_Repetition;
    env_parameter.BeamWidth_vertical = BeamWidth_vertical;
    env_parameter.K_bw = len(N_SSW_BS_vec);
    env_parameter.K_tslot = len(t_slot_vec);
    env_parameter.single_side_beam_training = single_side_beam_training;
    Coeff_BS2UE = np.zeros((env_parameter.K_bw,env_parameter.K_tslot))
    for i in range(env_parameter.K_tslot):
        t_slot = t_slot_vec[i];
        if single_side_beam_training:            
            n_sweep = (N_SSW_BS_vec+N_SSW_UE_vec)*Beampair_Repetition*ratio_codebook_used/(Num_arr_AP*Num_arr_UE)
        else:
            n_sweep = (N_SSW_BS_vec*N_SSW_UE_vec)*Beampair_Repetition*ratio_codebook_used/(Num_arr_AP*Num_arr_UE)
        Coeff_BS2UE[:,i] = 1 - t_SSW / t_slot * np.ceil(n_sweep)
    Coeff_BS2UE[np.where(Coeff_BS2UE<0)] = 0;
    env_parameter.ratio_codebook_used = ratio_codebook_used;
    env_parameter.Coeff_BS2UE = Coeff_BS2UE;
    env_parameter.channel_corr_coeff = channel_corr_coeff;
    env_parameter.mean_X = mean_X;
    env_parameter.sigma_X = sigma_X;
    env_parameter.min_use_per_bw_guaranteed = 1;
    env_parameter.min_use_per_tslot_guaranteed = 1;
    env_parameter.min_use_per_bw_tslot_guaranteed = 1;
    env_parameter.min_use_per_beam_guaranteed = 1;

    
    # -------------------------------------
    # Saved Mobility model
    # -------------------------------------
    # Position
    env_parameter.radius = radius
    env_parameter.Xcoor_init = Xcoor_init;
    env_parameter.Ycoor_init = Ycoor_init;
    env_parameter.Xcoor_list = [list() for _ in range(len(Xcoor_init))]
    env_parameter.Ycoor_list = [list() for _ in range(len(Xcoor_init))]
    for u in range(len(Xcoor_init)):
        env_parameter.Xcoor_list[u].append(Xcoor_init[u]);
        env_parameter.Ycoor_list[u].append(Ycoor_init[u]);
    env_parameter.Xcoor = Xcoor_init;
    env_parameter.Ycoor = Ycoor_init;
    # Velocity
    env_parameter.max_activity_range = max_activity_range
    env_parameter.v_min = v_min
    env_parameter.v_max = v_max
    env_parameter.number_last_direction = number_last_direction
    env_parameter.last_direction = last_direction
    env_parameter.last_velocity = last_velocity
    # Self-rotation
    env_parameter.v_self_rotate_min = v_self_rotate_min
    env_parameter.v_self_rotate_max = v_self_rotate_max
    env_parameter.number_last_rotate = number_last_rotate
    env_parameter.max_number_last_rotate = max_number_last_rotate
    env_parameter.max_number_last_direction = max_number_last_direction
    
    
    # -------------------------------------
    # Saved Blockage model
    # -------------------------------------s
    env_parameter.blockage_loss_dB = blockage_loss_dB;    # Blockage
    env_parameter.target_prob_blockage = target_prob_blockage;
    env_parameter.prob_blockage = prob_blockage;    # Blockage
    env_parameter.min_blockage_duration = min_blockage_duration; # Number of minimum slots that blockage exists
    env_parameter.max_blockage_duration = max_blockage_duration; # Number of maximum slots that blockage exists
    env_parameter.min_blockage_duration_guess = min_blockage_duration_guess;
    env_parameter.max_blockage_duration_guess = max_blockage_duration_guess;
    env_parameter.outage_coeff = np.zeros((env_parameter.N_UE,env_parameter.N_UE))
    env_parameter.last_outage_coeff = np.zeros((env_parameter.N_UE,env_parameter.N_UE))
    return env_parameter


#-----------------------------------------------------------------------
# In the following class, we create a enviroment
#-----------------------------------------------------------------------
class envs():
    # -------------------------------------
    # Initialization
    # -------------------------------------
    def __init__(self,env_parameter,slots_monitored):
        self.env_parameter = copy.deepcopy(env_parameter)
        self.slots_monitored = slots_monitored
        # Following would be reset when reset() called
        self.ct = 0
        self.remain_slots_in_blockage = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)
        self.num_slots_to_last_blockage_starts = np.ones(env_parameter.N_UE,dtype=int) * 100
        self.pathloss_history = np.zeros((slots_monitored,env_parameter.N_UE,env_parameter.N_UE))
        self.outage_coeff = np.zeros((env_parameter.N_UE,env_parameter.N_UE))
        self.position_update(def_action(env_parameter))
        self.pathloss_update()
        self.channel_X_history = np.zeros((slots_monitored,env_parameter.N_UE,env_parameter.N_UE))
        self.throughput = np.zeros((slots_monitored,env_parameter.N_UE));
        self.is_in_blockage = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)

    # -------------------------------------    
    # Reset enviroment
    # -------------------------------------
    def reset(self):
        self.ct = 0
        self.remain_slots_in_blockage = np.zeros((self.env_parameter.N_UE,self.env_parameter.N_UE),dtype=int)
        self.num_slots_to_last_blockage_starts = np.ones(self.env_parameter.N_UE,dtype=int) * 100
        self.pathloss_history = np.zeros((self.slots_monitored,self.env_parameter.N_UE,self.env_parameter.N_UE))
        self.outage_coeff = np.zeros((self.env_parameter.N_UE,self.env_parameter.N_UE))
        self.position_update(def_action(self.env_parameter))
        self.pathloss_update()
        self.channel_X_history = np.zeros((self.slots_monitored,self.env_parameter.N_UE,self.env_parameter.N_UE))
        self.throughput = np.zeros((self.slots_monitored,self.env_parameter.N_UE))
        self.is_in_blockage = np.zeros((self.env_parameter.N_UE,self.env_parameter.N_UE),dtype=int)
        
    # -------------------------------------
    # Run the enviroment for one time slot
    # -------------------------------------
    def step(self, state, action):
        # Get action
        bw_id = action.bw_id;
        tslot_id = action.tslot_id;
        beam_id = action.beam_id; 

        # Enviroment parameter
        N_SSW_BS_vec = self.env_parameter.N_SSW_BS_vec;
        N_SSW_UE_vec = self.env_parameter.N_SSW_UE_vec;
        W = self.env_parameter.W;
        B = self.env_parameter.B;
        t_slot_vec = self.env_parameter.t_slot_vec;
        t_slot = t_slot_vec[action.tslot_id];
        t_SSW = self.env_parameter.t_SSW;
        Ptx_BS_dBm = self.env_parameter.Ptx_BS_dBm;    
        GtGr_dBi_mat = self.env_parameter.GtGr_dBi_mat;
        Pn_dBm = self.env_parameter.Pn_dBm;
        SNR_Value = self.env_parameter.SNR_Value;
        Rate_Value = self.env_parameter.Rate_Value;
        Using_MCS = self.env_parameter.Using_MCS;
        Pathloss = self.env_parameter.Pathloss; 

        # Get Channel
        Relay_ID = 0;
        UE_ID_BS2UE_Link = Relay_ID;
        channel = self.channel_realization();
        X_BS2UE = channel[Relay_ID,Relay_ID];

        # Update the UE position (Mobility modeling)
        self.position_update(action)
        
        # Update the pathloss (Blockage modeling)
        self.pathloss_update()
        
        # Update outage event (Outage modeling)
        self.outage_coeff_update(action)
        
        # Beam training and data transmission for main link
        SNR_BS2UE_dB = Ptx_BS_dBm + GtGr_dBi_mat[bw_id] - Pathloss[UE_ID_BS2UE_Link,UE_ID_BS2UE_Link] - W - Pn_dBm + X_BS2UE;
        
        # Unknown Coeff_BS2UE
        Coeff_BS2UE = self.env_parameter.Coeff_BS2UE[bw_id,tslot_id];
        Coeff_BS2UE = Coeff_BS2UE * (1-self.outage_coeff[UE_ID_BS2UE_Link,UE_ID_BS2UE_Link]);
        if Using_MCS:
            MCS_ID_BS2UE = len(np.where(SNR_Value<=SNR_BS2UE_dB)[0])-1;
            Reff_BS2UE_Link = Coeff_BS2UE*Rate_Value[MCS_ID_BS2UE];
        else:
            Reff_BS2UE_Link = Coeff_BS2UE*B*np.log2(1+10^(SNR_BS2UE_dB/10));
        Reff_BS2UE_Link = max(Reff_BS2UE_Link,0);
        
        # Output        
        reward = Reff_BS2UE_Link;
        MCS_level = MCS_ID_BS2UE;
        
        self.ct += 1
        self.num_slots_to_last_blockage_starts += 1  
        
        # Return variables
        return reward, MCS_level

    
    # -------------------------------------
    # Channel shadowing realization
    # -------------------------------------
    def channel_realization(self):
        for u1 in range(self.env_parameter.N_UE):
            for u2 in range(u1,self.env_parameter.N_UE):
                self.channel_X_history[self.ct,u1,u2] = np.random.normal(self.env_parameter.mean_X[u1,u2],self.env_parameter.sigma_X[u1,u2])
                self.channel_X_history[self.ct,u2,u1] = self.channel_X_history[self.ct,u1,u2]
        channel = np.squeeze(self.channel_X_history[self.ct,:,:])
        return channel
    
    # -------------------------------------
    # Position update with user mobility
    # -------------------------------------
    def position_update(self,action):
        # fetch slot duration
        t_slot = self.env_parameter.t_slot_vec[action.tslot_id]
        # Compute new position
        if self.ct == 0:
            self.env_parameter.Xcoor = self.env_parameter.Xcoor_init;
            self.env_parameter.Ycoor = self.env_parameter.Ycoor_init;
            self.env_parameter.Xcoor_list = [list() for _ in range(self.env_parameter.N_UE+1)]
            self.env_parameter.Ycoor_list = [list() for _ in range(self.env_parameter.N_UE+1)]
            for u in range(self.env_parameter.N_UE+1):
                self.env_parameter.Xcoor_list[u].append(self.env_parameter.Xcoor_init[u]);
                self.env_parameter.Ycoor_list[u].append(self.env_parameter.Ycoor_init[u]);
            #self.env_parameter.number_last_direction = np.zeros(self.env_parameter.N_UE+1)   
            self.env_parameter.direction_new = self.env_parameter.last_direction
            self.env_parameter.velocity_new = self.env_parameter.last_velocity
        else:
            # Fetch initial position
            Xcoor_init = self.env_parameter.Xcoor_init
            Ycoor_init = self.env_parameter.Ycoor_init
            # Fetech current position
            Xcoor = self.env_parameter.Xcoor
            Ycoor = self.env_parameter.Ycoor
            # Compute the new position
            Xcoor_new = np.zeros(self.env_parameter.N_UE+1)
            Ycoor_new = np.zeros(self.env_parameter.N_UE+1)
            # Current distance to the initial position
            current_dist_to_original = np.sqrt((Xcoor-Xcoor_init)**2 + (Ycoor-Ycoor_init)**2)  
            new_dist_to_original = np.zeros(self.env_parameter.N_UE+1)
            direction_new = np.zeros(self.env_parameter.N_UE+1)
            velocity_new = np.zeros(self.env_parameter.N_UE+1)
            for u in range(self.env_parameter.N_UE+1):
                # Case UE is currently on border
                if abs(current_dist_to_original[u]-self.env_parameter.max_activity_range) <= 1e-6:
                    # Bouncing back to the original position
                    direction_new[u] = np.arctan2(Ycoor[u]-Ycoor_init[u],Xcoor[u]-Xcoor_init[u]) + np.pi 
                    velocity_new[u] = self.env_parameter.last_velocity[u]
                    #print('Bouncing back')
                else:
                    if self.env_parameter.number_last_direction[u] >= self.env_parameter.max_number_last_direction:
                        direction_new[u] = np.random.uniform(low=-np.pi, high=np.pi,size=1)
                        velocity_new[u] = np.random.uniform(low=self.env_parameter.v_min[u], high=self.env_parameter.v_max[u], size=1)
                        self.env_parameter.number_last_direction[u] = 0
                        #print('New random direction')
                    else:
                        direction_new[u] = self.env_parameter.last_direction[u]
                        velocity_new[u] = self.env_parameter.last_velocity[u]
                        self.env_parameter.number_last_direction[u] += 1
                        #print('Continue the previous direction')
                self.env_parameter.last_direction[u] = direction_new[u]
                self.env_parameter.last_velocity[u] = velocity_new[u]
                Xcoor_new[u] = Xcoor[u] + np.cos(direction_new[u]) * velocity_new[u] * t_slot
                Ycoor_new[u] = Ycoor[u] + np.sin(direction_new[u]) * velocity_new[u] * t_slot                
                new_dist_to_original[u] = np.sqrt((Xcoor_new[u]-Xcoor_init[u])**2 + (Ycoor_new[u]-Ycoor_init[u])**2)
                if  new_dist_to_original[u] > self.env_parameter.max_activity_range:
                    Xcoor_new[u] = Xcoor_init[u] + self.env_parameter.max_activity_range * 0.9 *\
                    np.cos(np.arctan2(Ycoor_new[u]-Ycoor_init[u],Xcoor_new[u]-Xcoor_init[u]))
                    Ycoor_new[u] = Ycoor_init[u] + self.env_parameter.max_activity_range * 0.9 *\
                    np.sin(np.arctan2(Ycoor_new[u]-Ycoor_init[u],Xcoor_new[u]-Xcoor_init[u]))
                # Save historical position
                self.env_parameter.Xcoor_list[u].append(Xcoor_new[u]);
                self.env_parameter.Ycoor_list[u].append(Ycoor_new[u]);
            # Update new coordinates and speed
            self.env_parameter.direction_new = direction_new
            self.env_parameter.velocity_new = velocity_new
            self.env_parameter.Xcoor = Xcoor_new
            self.env_parameter.Ycoor = Ycoor_new    
            
            # Sanity check of new coordiantes
            new_dist_to_original_check = np.sqrt((self.env_parameter.Xcoor_init - self.env_parameter.Xcoor)**2 + \
                                           (self.env_parameter.Ycoor_init - self.env_parameter.Ycoor)**2)   
            if any((new_dist_to_original_check-self.env_parameter.max_activity_range) > 1e-1):
                pdb.set_trace()
                sys.exit('Ue moves out of the activity region')
    
    # -------------------------------------
    # Pathloss update in dB with UMa model and blockage model
    # -------------------------------------
    def pathloss_update(self):
        # Compute new distance
        dist_D2D = np.ones((self.env_parameter.N_UE,self.env_parameter.N_UE));
        for u1 in range(self.env_parameter.N_UE):
            dist_D2D[u1,u1] = np.sqrt((self.env_parameter.Xcoor[u1]-self.env_parameter.Xcoor[-1])**2 +\
                                      (self.env_parameter.Ycoor[u1]-self.env_parameter.Ycoor[-1])**2);
            for u2 in range(u1+1,self.env_parameter.N_UE,1):
                dist_D2D[u1,u2] = np.sqrt((self.env_parameter.Xcoor[u1]-self.env_parameter.Xcoor[u2])**2 +\
                                          (self.env_parameter.Ycoor[u1]-self.env_parameter.Ycoor[u2])**2);
                dist_D2D[u2,u1] = dist_D2D[u1,u2];
        # Blocakge
        if self.ct ==0:
            Pathloss = 28.0 + 22*np.log10(dist_D2D) + 20*np.log10(self.env_parameter.fc);
            self.env_parameter.Pathloss = Pathloss
        else:
            new_blockage_duration = \
            np.random.randint(self.env_parameter.min_blockage_duration,self.env_parameter.max_blockage_duration+1,\
                             (self.env_parameter.N_UE,self.env_parameter.N_UE))
            new_blockage_status = np.random.binomial(1,self.env_parameter.prob_blockage) 
            for uu1 in range(self.env_parameter.N_UE):
                for uu2 in range(uu1+1,self.env_parameter.N_UE):
                    new_blockage_duration[uu2,uu1] = new_blockage_duration[uu1,uu2];
                    new_blockage_status[uu2,uu1] = new_blockage_status[uu1,uu2];
                    
            self.remain_slots_in_blockage -= 1;
            self.remain_slots_in_blockage.clip(min=0,out=self.remain_slots_in_blockage)            
            is_in_blockage = np.zeros_like(self.remain_slots_in_blockage)
            is_in_blockage.clip(max=1,out=is_in_blockage)           
            self.remain_slots_in_blockage = self.remain_slots_in_blockage + \
            (1 - is_in_blockage) * new_blockage_duration * new_blockage_status
            is_in_blockage = np.zeros_like(self.remain_slots_in_blockage)
            self.remain_slots_in_blockage.clip(max=1,out=is_in_blockage)     
            self.is_in_blockage = is_in_blockage
            blockage_loss_dB = is_in_blockage * self.env_parameter.blockage_loss_dB
            Pathloss = 28.0 + 22*np.log10(dist_D2D) + 20*np.log10(self.env_parameter.fc) + blockage_loss_dB; 
            self.env_parameter.Pathloss = Pathloss
            self.num_slots_to_last_blockage_starts = self.num_slots_to_last_blockage_starts * \
            (1-np.diag(new_blockage_status))
        
        # Save path loss history
        self.pathloss_history[self.ct,:,:] = Pathloss

    # -------------------------------------    
    # Outage duration calculatio for a successfully connected
    # The event outage is a function of UE distance, UE velocity (value and direction), 
    # UE self-rotation, time slot duration, and beamwidth
    # -------------------------------------
    def outage_coeff_update(self, action):
        # fetch slot duration
        t_slot = self.env_parameter.t_slot_vec[action.tslot_id]
        self.env_parameter.last_outage_coeff = self.env_parameter.outage_coeff
        outage_coeff = 0
        if self.ct == 0:
            return outage_coeff
        # Fetch mobility parameters
        direction_new = self.env_parameter.direction_new
        v_tmp = self.env_parameter.velocity_new
        v_self_rotate_min = self.env_parameter.v_self_rotate_min
        v_self_rotate_max = self.env_parameter.v_self_rotate_max
        self.env_parameter.v_self_rotate = np.random.uniform(low=v_self_rotate_min,high=v_self_rotate_max,size=self.env_parameter.N_UE)
        if self.env_parameter.number_last_rotate[0] <= self.env_parameter.max_number_last_rotate:
            self.env_parameter.number_last_rotate += 1
        else:
            self.env_parameter.v_self_rotate = self.env_parameter.v_self_rotate *(2*np.random.binomial(1,0.5,self.env_parameter.N_UE)-1)
            self.env_parameter.number_last_rotate = np.zeros_like(self.env_parameter.number_last_rotate);
        v_self_rotate = self.env_parameter.v_self_rotate
        num_seg_slot = 1000
        t_seg_vec = np.asarray(range(1,num_seg_slot+1))*t_slot/num_seg_slot
        
        # Outage in main link for case l=0
        # Evolution of Tx mobility
        uTx = 0
        uRx = uTx
        v_tmp_Tx = v_tmp[-1]
        Xcoor_Tx_last = self.env_parameter.Xcoor_list[-1][-2]
        Ycoor_Tx_last = self.env_parameter.Ycoor_list[-1][-2]
        Xcoor_Tx_evo = Xcoor_Tx_last + np.cos(direction_new[-1]) * v_tmp_Tx * t_seg_vec
        Ycoor_Tx_evo = Ycoor_Tx_last + np.sin(direction_new[-1]) * v_tmp_Tx * t_seg_vec
        theta_Tx_self_rotate = v_self_rotate[-1] * t_seg_vec * 0 # Set Tx not rotating
        theta_Rx_self_rotate = v_self_rotate[uRx] * t_seg_vec * 0
        # Evolution of Rx mobility
        v_tmp_Rx = v_tmp[uRx]
        Xcoor_Rx_last = self.env_parameter.Xcoor_list[uRx][-2]
        Ycoor_Rx_last = self.env_parameter.Ycoor_list[uRx][-2]
        Xcoor_Rx_evo = Xcoor_Rx_last + np.cos(direction_new[uRx]) * v_tmp_Rx * t_seg_vec
        Ycoor_Rx_evo = Ycoor_Rx_last + np.sin(direction_new[uRx]) * v_tmp_Rx * t_seg_vec
        # Calculate the angle due the the Rx self-rotation
        vector_Tx_point_to_Rx_Xcoor =  Xcoor_Rx_last - Xcoor_Tx_last
        vector_Tx_point_to_Rx_Ycoor =  Ycoor_Rx_last - Ycoor_Tx_last
        vector_Txevo_point_to_Rxevo_Xcoor = Xcoor_Rx_evo - Xcoor_Tx_evo
        vector_Txevo_point_to_Rxevo_Ycoor = Ycoor_Rx_evo - Ycoor_Tx_evo
        tmp_cos_value = (vector_Tx_point_to_Rx_Xcoor * vector_Txevo_point_to_Rxevo_Xcoor+\
                         vector_Tx_point_to_Rx_Ycoor * vector_Txevo_point_to_Rxevo_Ycoor)/\
        (np.sqrt(vector_Tx_point_to_Rx_Xcoor**2+vector_Tx_point_to_Rx_Ycoor**2)*\
         np.sqrt(vector_Txevo_point_to_Rxevo_Xcoor**2+vector_Txevo_point_to_Rxevo_Ycoor**2))
        tmp_cos_value[np.where(tmp_cos_value>1)[0]] = 1
        intersection_angle_point_to_new_Rx = np.degrees(np.arccos(tmp_cos_value))
        intersection_angle_point_to_new_Tx = intersection_angle_point_to_new_Rx
        # Check outage and calculate outage coefficent
        bw_Tx = self.env_parameter.BeamWidth_TX_vec[action.bw_id]
        bw_Rx = self.env_parameter.BeamWidth_RX_vec
        outage_in_Tx_beam = bw_Tx/2 - intersection_angle_point_to_new_Rx - theta_Tx_self_rotate
        outage_in_Rx_beam = bw_Rx/2 - intersection_angle_point_to_new_Tx - theta_Rx_self_rotate
        outage_in_Tx_beam[np.where(outage_in_Tx_beam>=0)] = 1
        outage_in_Tx_beam[np.where(outage_in_Tx_beam<0)] = 0
        outage_in_Rx_beam[np.where(outage_in_Rx_beam>=0)] = 1
        outage_in_Rx_beam[np.where(outage_in_Rx_beam<0)] = 0
        violation_status = outage_in_Tx_beam * outage_in_Rx_beam
        violation_slots = np.where(violation_status==0)[0]
        if len(violation_slots) >=1:
            outage_coeff = 1 - violation_slots[0]/num_seg_slot
        else:
            outage_coeff = 0
        self.outage_coeff[uTx,uRx] = outage_coeff
        self.env_parameter.outage_coeff = self.outage_coeff        
        return outage_coeff