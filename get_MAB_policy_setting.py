class def_policy_setting:
    def __init__(self):
        self.enable_bandit = False;
        self.enable_TS = False;
        self.enable_random = False;
        self.enable_correlation = False;
        self.enable_monte_carlo = False;
        self.policy_id = -1;
        self.legend = 'Default';
        self.color = 'r';
        self.default_bw_id = 4;
        self.default_tslot_id = 1;
        self.default_beam_id = -1;
        self.ratio_codebook_used = 1;
        return
        
def get_MAB_policy_setting(exid,env_parameter):
    policy_setting_list = list();
    color_arr = ['C1','C4','C5','C6','C7','C8','C9'] 
    # Beamwidth selection (K_bw + 2 polices)
    if exid == 1:
        # All fixed policy
        for i in range(env_parameter.K_bw):
            policy_setting = def_policy_setting();
            policy_setting.policy_id = i;
            policy_setting.default_bw_id = i;
            policy_setting.legend = 'Fixed beamwidth level: ' + str(i);
            policy_setting.color = color_arr[i];
            policy_setting_list.append(policy_setting);
        # Random policy
        policy_setting = def_policy_setting();
        policy_setting.enable_random = True;
        policy_setting.policy_id = env_parameter.K_bw;
        policy_setting.legend = 'Random selection policy';
        policy_setting.color = 'C0';
        policy_setting_list.append(policy_setting);
        # Bandit policy
        policy_setting = def_policy_setting();
        policy_setting.enable_bandit = True;
        policy_setting.policy_id = env_parameter.K_bw+1;
        policy_setting.legend = 'Bandit (TS) policy';
        policy_setting.color = 'C3';
        policy_setting_list.append(policy_setting);
        
    # Beam sweeping period selection (K_tslot + 2 polices)
    elif exid == 2:
        # All fixed policy
        for i in range(env_parameter.K_tslot):
            policy_setting = def_policy_setting();
            policy_setting.policy_id = i;
            policy_setting.default_tslot_id = i;
            policy_setting.legend = 'Fixed beam sweeping periodicity: ' + str(env_parameter.t_slot_vec[i]*1000)+' ms';
            policy_setting.color = color_arr[i];
            policy_setting_list.append(policy_setting);
        # Random policy
        policy_setting = def_policy_setting();
        policy_setting.enable_random = True;
        policy_setting.enable_bandit = False;
        policy_setting.enable_TS = False;
        policy_setting.policy_id = env_parameter.K_tslot;
        policy_setting.legend = 'Random selection policy';
        policy_setting.color = 'C0';
        policy_setting_list.append(policy_setting);
        # Bandit policy
        policy_setting = def_policy_setting();
        policy_setting.enable_random = False;
        policy_setting.enable_bandit = True;
        policy_setting.enable_TS = True;
        policy_setting.policy_id = env_parameter.K_tslot+1;
        policy_setting.legend = 'Bandit (TS) policy';
        policy_setting.color = 'C2';
        policy_setting_list.append(policy_setting);
        # Bandit policy
        policy_setting = def_policy_setting();
        policy_setting.enable_random = False;
        policy_setting.enable_bandit = True;
        policy_setting.enable_TS = False;
        policy_setting.policy_id = env_parameter.K_tslot+2;
        policy_setting.legend = 'Bandit (KL-UCB) policy';
        policy_setting.color = 'C3';
        policy_setting_list.append(policy_setting);
        
    # Joint Beamwidth and Beam sweeping periodicity selection (K_tslot * K_bw + 3 polices)
    elif exid == 3:
        policy_id_count = 0;
        # All fixed policy
        for i in range(env_parameter.K_bw):
            for j in range(env_parameter.K_tslot):
                policy_setting = def_policy_setting();
                policy_setting.policy_id = i*env_parameter.K_tslot + j;
                policy_setting.legend = 'Fixed beam sweeping periodicity: ' + str(env_parameter.t_slot_vec[j]*1000)+' ms and fixed beamwidth level: ' +str(i) ;
                idc = policy_setting.policy_id % len(color_arr)
                policy_setting.color = color_arr[idc];
                policy_setting.default_bw_id = i;
                policy_setting.default_tslot_id = j;
                policy_setting_list.append(policy_setting);
                policy_id_count = policy_id_count + 1;
        # Random policy
        policy_setting = def_policy_setting();
        policy_setting.enable_random = True;
        policy_setting.policy_id = policy_id_count;
        policy_setting.legend = 'Random selection policy';
        policy_setting.color = 'C0';
        policy_setting_list.append(policy_setting);
        policy_id_count = policy_id_count + 1;
        # Bandit policy
        policy_setting = def_policy_setting();
        policy_setting.enable_bandit = True;
        policy_setting.policy_id = policy_id_count;
        policy_setting.legend = 'Bandit (KL-UCB) policy';
        policy_setting.color = 'C2';
        policy_setting_list.append(policy_setting);
        policy_id_count = policy_id_count + 1;
        # Bandit policy
        policy_setting = def_policy_setting();
        policy_setting.enable_bandit = True;
        policy_setting.enable_monte_carlo = True;
        policy_setting.policy_id = policy_id_count;
        policy_setting.legend = 'Bandit (KL-UCB) based Monte-Carlo policy';
        policy_setting.color = 'C3';
        policy_setting_list.append(policy_setting);
        policy_id_count = policy_id_count + 1;
        
    # Joint beamwidth, beam training peeriod and beam direction selection (6 policies)
    else: # exid == 5
        # All beams with random codebook policy
        policy_setting = def_policy_setting();
        policy_setting.enable_random = True;
        policy_setting.policy_id = 0;
        policy_setting.ratio_codebook_used = 1;
        policy_setting.legend = 'Random selection over beamwidth and beam sweeping periodicity with all beams';
        policy_setting.color = 'C0';
        policy_setting_list.append(policy_setting);
        # Bandit policy with R = 1/4
        policy_setting = def_policy_setting();
        policy_setting.enable_bandit = True;
        policy_setting.ratio_codebook_used = 1/1;
        policy_setting.policy_id = 1;
        policy_setting.legend = 'Bandit (KL-UCB) based Monte-Carlo policy with all beams';
        policy_setting.color = 'C1';
        policy_setting_list.append(policy_setting);
        # Bandit policy with R = 1/2
        policy_setting = def_policy_setting();
        policy_setting.enable_bandit = True;
        policy_setting.ratio_codebook_used = 1/2;
        policy_setting.policy_id = 2;
        policy_setting.legend = 'Bandit (KL-UCB) based Monte-Carlo policy selecting 1/2 of all beams';
        policy_setting.color = 'C2';
        policy_setting_list.append(policy_setting);
        # Bandit policy with R = 1/4
        policy_setting = def_policy_setting();
        policy_setting.enable_bandit = True;
        policy_setting.ratio_codebook_used = 1/4;
        policy_setting.policy_id = 3;
        policy_setting.legend = 'Bandit (KL-UCB) based Monte-Carlo policy selecting 1/4 of all beams';
        policy_setting.color = 'C3';
        policy_setting_list.append(policy_setting);
    return policy_setting_list