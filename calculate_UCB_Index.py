import numpy as np
import math
import pdb

def calculate_UCB_Index(enable_correlation,success,Mean_Beam,NumUse_beam,UCB_Index_beam,i_sel,j_sel,I,r,M,K,ct):
    epsilon = 0.00001;
    c = 0;
    t = ct+1;
    if t == 1:
        t = t + epsilon;
    ft_KLUCB = np.log(t) + c*np.log(np.log(t));
    ft_UCB = t**(1.5);
    # Update mean, num and ucb_index
    ft = ft_KLUCB;    
    Mean_Beam[i_sel,j_sel] = (Mean_Beam[i_sel,j_sel]*NumUse_beam[i_sel,j_sel] + success[i_sel,j_sel])/(NumUse_beam[i_sel,j_sel]+1);
    NumUse_beam[i_sel,j_sel] = NumUse_beam[i_sel,j_sel] + 1;
    for j in range(M[i_sel]):
        if NumUse_beam[i_sel,j] >= 1:
            UCB_Index_beam[i_sel,j] = findKLUCB(Mean_Beam[i_sel,j],ft,NumUse_beam[i_sel,j],epsilon);             
    if enable_correlation: # KL-UCB deterministic propagation
        success_est = np.full_like(success, np.nan);
        for j_ind in range(len(j_sel)):
            j = j_sel[j_ind];
            # Up propagation
            if success[i_sel,j] == 1:
                k = j
                for i in range(i_sel-1,-1,-1):
                    #rr = np.prod(r[i+1:i_sel+1]);
                    rr = r[i+1]
                    k = k // rr
                    success_est[i,k] = success[i,k]
                    Mean_Beam[i,k] = (Mean_Beam[i,k]*NumUse_beam[i,k] + success_est[i,k])/(NumUse_beam[i,k]+1);
                    NumUse_beam[i,k] = NumUse_beam[i,k] + 1;
                    UCB_Index_beam[i,k] = findKLUCB(Mean_Beam[i,k],ft,NumUse_beam[i,k],epsilon);
            # Down propagation
            elif success[i_sel,j] == 0:
                for i in range(i_sel+1,I,1):
                    rr = np.prod(r[i_sel+1:i+1])
                    for k in range(j*rr,(j+1)*rr):
                        success_est[i,k] = success[i,k]
                        Mean_Beam[i,k] = (Mean_Beam[i,k]*NumUse_beam[i,k] + success_est[i,k])/(NumUse_beam[i,k]+1);
                        NumUse_beam[i,k] = NumUse_beam[i,k] + 1;
                        UCB_Index_beam[i,k] = findKLUCB(Mean_Beam[i,k],ft,NumUse_beam[i,k],epsilon);
            else:
                pass;
    return Mean_Beam, NumUse_beam, UCB_Index_beam


def findKLUCB(Mean_arm,ft,Num_arm_use,epsilon):
    if Mean_arm == 1:
        Mean_arm = 1 - epsilon;
    elif Mean_arm == 0:
        Mean_arm = epsilon;
    if ft == 0 and Num_arm_use == 1:
        q_ucb = Mean_arm;
        return q_ucb
    low = Mean_arm;
    up = 1 - epsilon/10;
    middle = (up+low)/2;
#     try:    
    threshold = np.log(ft)/Num_arm_use;
#     except RuntimeWarning:
#         pdb.set_trace()
    dpq_v = dpq(Mean_arm,middle);
    resolution = epsilon/10;
    while abs(up-low)>resolution:
        if dpq_v > threshold:
            up = middle;
        else:
            low = middle;
        middle = (up+low)/2;
        dpq_v = dpq(Mean_arm,middle);
    q_ucb = middle;
    return q_ucb


def dpq(p,q):
    related_entropy = p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q));
    return related_entropy