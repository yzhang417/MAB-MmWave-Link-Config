# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb
import matplotlib.pyplot as plt



#-------------------------------------------------------------------------
# Transform cartesian coordinates to polar coordiantes 
#-------------------------------------------------------------------------
def pol2cart(angle,radius):
    x = radius*np.cos(angle)
    y = radius*np.sin(angle)
    return x,y


#-------------------------------------------------------------------------
# Plot network topology
#-------------------------------------------------------------------------
def plot_network_topology(env_parameter, output_folder, show_mobility_trace, show_plot, exid):
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'legend.fontsize': 14})
    plt.rcParams.update({'legend.loc': 'lower right'})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['lines.linewidth'] = 1.5
    ue_color = ['b','g','c','m','k','r']
    number_dots_border = range(360);
    Xcoor_border = env_parameter.max_activity_range * np.cos(np.radians(number_dots_border))
    Ycoor_border = env_parameter.max_activity_range * np.sin(np.radians(number_dots_border))            
    fig = plt.figure(figsize=(10,6),dpi=1200);   
    ax_netw_topo = fig.add_subplot(111)
    ax_netw_topo.axis('equal')
    ax_netw_topo.grid(b=True, which='major', color='#666666', linestyle='-')
    ax_netw_topo.plot(env_parameter.Xcoor_init[-1],env_parameter.Ycoor_init[-1],'s',label='Base station',c=ue_color[-1])
    for u in range(1):
        ax_netw_topo.plot(env_parameter.Xcoor_init[u],env_parameter.Ycoor_init[u],'*', label='Initial position of user',c=ue_color[u])
        ax_netw_topo.plot(env_parameter.Xcoor_init[u] + Xcoor_border,env_parameter.Ycoor_init[u] + Ycoor_border,'--', label='Cell boundary',c='k')
    #ax_netw_topo.plot(env_parameter.Xcoor_init[-1] + Xcoor_border,env_parameter.Ycoor_init[-1] + Ycoor_border,'-', label='Border of AP',c=ue_color[-1])
    ax_netw_topo.set_xlabel('X-axis (meters)')
    ax_netw_topo.set_ylabel('Y-axis (meters)')
    ax_netw_topo.set_title('Network topology')
    ax_netw_topo.legend(loc='lower right')
    if show_mobility_trace:
        for u in range(1):
            ax_netw_topo.plot(env_parameter.Xcoor_list[u],env_parameter.Ycoor_list[u],'-',c=ue_color[u],linewidth=0.3, markersize=0.15, label='One realization of moving trajectory')
        plt.savefig(output_folder+'Network_topology_trace_'+str(exid)+'.eps',format='eps', facecolor='w', transparent=False, dpi=1200)
    else:
        plt.savefig(output_folder+'Network_topology_'+str(exid)+'.eps',format='eps', facecolor='w', transparent=False, dpi=1200)
    if show_plot:
        plt.show()
    else:
        plt.close()
        
        
#-------------------------------------------------------------------------
# Plot evolution of rate
#-------------------------------------------------------------------------
def plot_result(env_parameter, evolution_reward, evolution_tslot, policy_setting_list, exid, output_folder):
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'legend.fontsize': 10})
    plt.rcParams.update({'legend.loc': 'lower right'})
    plt.rcParams.update({'figure.autolayout': True})        
    figID = exid*100
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    ave_evolution_reward = np.squeeze(np.mean(evolution_reward,axis=0))/1e9
    ave_reward = np.mean(ave_evolution_reward,axis=0)
    if exid == 3 or exid == 2:
        ave_reward_fixed_policy = ave_reward[:-3];
    else:
        ave_reward_fixed_policy = ave_reward[:-2];
    genius_policy_id = np.argmax(ave_reward_fixed_policy)
    worst_policy_id = np.argmin(ave_reward_fixed_policy)
    t_slot_vec =  env_parameter.t_slot_vec * 1000
    N_SSW_BS_vec = env_parameter.N_SSW_BS_vec
    if exid == 4 or exid == 5:
        policy_to_be_shown = range(len(policy_setting_list))
        #plt.rcParams.update({'legend.loc': 'lower right'})
    elif exid == 3:
        policy_to_be_shown = [genius_policy_id, worst_policy_id, len(policy_setting_list)-3, len(policy_setting_list)-2, len(policy_setting_list)-1];
        genius_tslot = t_slot_vec[int(policy_setting_list[genius_policy_id].default_tslot_id)]
        genius_bw = N_SSW_BS_vec[int(policy_setting_list[genius_policy_id].default_bw_id)]
        worst_tslot = t_slot_vec[int(policy_setting_list[worst_policy_id].default_tslot_id)]
        worst_bw = N_SSW_BS_vec[int(policy_setting_list[worst_policy_id].default_bw_id)]
        policy_setting_list[genius_policy_id].legend = 'Genius policy (beam sweeping period: ' + str(genius_tslot) + 'ms, number of sectors:' + str(genius_bw) + ')'
        policy_setting_list[worst_policy_id].legend = 'Worst policy (beam sweeping period: ' + str(worst_tslot) + 'ms, number of sectors:' + str(worst_bw) + ')'
    elif exid == 2:
        policy_to_be_shown = [genius_policy_id, worst_policy_id, len(policy_setting_list)-3, len(policy_setting_list)-2, len(policy_setting_list)-1];
        genius_tslot = t_slot_vec[int(policy_setting_list[genius_policy_id].default_tslot_id)]
        worst_tslot = t_slot_vec[int(policy_setting_list[worst_policy_id].default_tslot_id)]
        policy_setting_list[genius_policy_id].legend = 'Genius policy (beam sweeping period: ' + str(genius_tslot) + 'ms)'
        policy_setting_list[worst_policy_id].legend = 'Worst policy (beam sweeping period: ' + str(worst_tslot) + 'ms)'
    else:
        policy_to_be_shown = [genius_policy_id, worst_policy_id, len(policy_setting_list)-2, len(policy_setting_list)-1];
        genius_bw = N_SSW_BS_vec[int(policy_setting_list[genius_policy_id].default_bw_id)]
        worst_bw = N_SSW_BS_vec[int(policy_setting_list[worst_policy_id].default_bw_id)]
        policy_setting_list[genius_policy_id].legend = 'Genius policy (number of sectors:' + str(genius_bw) + ')'
        policy_setting_list[worst_policy_id].legend = 'Worst policy (number of sectors:' + str(worst_bw) +')'
    figID = exid*100
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    for i in policy_to_be_shown:
        policy_setting = policy_setting_list[i];
        plt.plot(range(len(ave_evolution_reward[1:,i])),ave_evolution_reward[1:,i],label=policy_setting_list[i].legend,c=policy_setting_list[i].color)
    plt.legend()
    plt.xlabel('Time slot index')
    plt.ylabel('Data rate (Gbps)');
    plt.savefig(output_folder+'example'+str(exid)+'.eps',format='eps', facecolor='w', transparent=False, dpi=1200)