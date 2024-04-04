import numpy as np
from scipy.optimize import curve_fit 
from llranalysis import llr
import pandas as pd
import os

def double_Gaussian_velocity(x, amp1, mn1, std1,amp2, mn2, std2):
    G1 = np.exp(-((x-mn1)**2.)/(2 * (std1**2.)))  * amp1
    G2 = np.exp(-((x-mn2)**2.)/(2 * (std2**2.))) * amp2
    return G1*((mn1 - x) / std1**2.) + G2*((mn2 - x) / std2**2.)

def double_Gaussian_acceleration(x, amp1, mn1, std1,amp2, mn2, std2):
    G1 = np.exp(-((x-mn1)**2.)/(2 * (std1**2.)))  * amp1
    G2 = np.exp(-((x-mn2)**2.)/(2 * (std2**2.))) * amp2
    return (G1/ std1**2.)*(((mn1 - x)/ std1)**2. -1. ) +  (G2/ std2**2.)*(((mn2 - x)/ std2)**2. -1. )

def double_Gaussian_find_peaks(x, amp1, mn1, std1,amp2, mn2, std2):
    G1 = np.exp(-((x-mn1)**2.)/(2 * (std1**2.)))  * amp1
    G2 = np.exp(-((x-mn2)**2.)/(2 * (std2**2.))) * amp2
    y_vel = G1*((mn1 - x) / std1**2.) + G2*((mn2 - x) / std2**2.)
    y_acc = (G1/ std1**2.)*(((mn1 - x)/ std1)**2. -1. ) +  (G2/ std2**2.)*(((mn2 - x)/ std2)**2. -1. )
    x_mid = x[y_acc > 0][np.argmin(np.abs(y_vel[y_acc > 0]))]
    y_mid = (G1+G2)[y_acc > 0][np.argmin(np.abs(y_vel[y_acc > 0]))]
    indx = [np.argmin(np.abs(y_vel[(y_acc < 0)*(x < x_mid)])),np.argmin(np.abs(y_vel[(y_acc < 0)*(x > x_mid)]))]    
    y = np.array([(G1 + G2)[(y_acc < 0)*(x < x_mid)][indx[0]],(G1 + G2)[(y_acc < 0)*(x > x_mid)][indx[1]],y_mid ])
    x = np.array([x[(y_acc < 0)*(x < x_mid)][indx[0]], x[(y_acc < 0)*(x > x_mid)][indx[1]], x_mid])
    return x, y

def double_Gaussian(x, amp1, mn1, std1,amp2, mn2, std2):
    G1 = np.exp(-((x-mn1)**2.)/(2 * (std1**2.)))  * amp1
    G2 = np.exp(-((x-mn2)**2.)/(2 * (std2**2.))) * amp2
    return G1 + G2

def find_critical_point(betas, dB, tol,N, final_df, full_obs,folder):
    Bc = full_obs['b'].values[np.argmax(full_obs['Cu'])]
    V = 6*final_df['V'].values[0]
    try:
        lnz = llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, Bc)
        x, y = llr.calc_prob_distribution(final_df, Bc, lnz)
        peaks = np.where(y > (0.5 * max(y)))
        mn1 = x[np.argmax(y)]
        mn2 = x[y > (0.5 * max(y))].mean() + (x[y > (0.5 * max(y))].mean() - mn1)
        [mn1,mn2] = [min([mn1,mn2]), max([mn1,mn2])]
        std = (mn2 - mn1) / 4
        estimates = (max(y),mn2,std,max(y),mn1,std)
        xs_tmp = np.linspace(np.min(x[y > ((0.5 * (max(y) / N)))]) , np.max(x[y > ((0.5 * (max(y) / N)))]), 1000)
        x, y = llr.calc_prob_distribution(final_df, Bc, lnz, xs_tmp)
        xopt, xcov = curve_fit(double_Gaussian, x, y, p0=estimates)
    except: 
        print('Error in fitting using alternate B_c')
        #Bc -= dB
        Bc = full_obs['b'].values[np.argmax(full_obs['Xlp'])]
        lnz = llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, Bc)
        x, y = llr.calc_prob_distribution(final_df, Bc, lnz)
        peaks = np.where(y > (0.5 * max(y)))
        mn1 = x[np.argmax(y)]
        mn2 = x[y > (0.5 * max(y))].mean() + (x[y > (0.5 * max(y))].mean() - mn1)
        [mn1,mn2] = [min([mn1,mn2]), max([mn1,mn2])]
        std = (mn2 - mn1) / 8
        estimates = (max(y),mn2,std,max(y),mn1,std)
        xs_tmp = np.linspace(np.min(x[y > ((0.5 * (max(y) / N)))]) , np.max(x[y > ((0.5 * (max(y) / N)))]), 1000)
        x, y = llr.calc_prob_distribution(final_df, Bc, lnz, xs_tmp)
        xopt, xcov = curve_fit(double_Gaussian, x, y, p0=estimates)
    j = 0.
    ys_tmp = double_Gaussian(xs_tmp, xopt[0], xopt[1], xopt[2], xopt[3], xopt[4], xopt[5])
    x, y = llr.calc_prob_distribution(final_df, Bc, lnz)
    b,a = double_Gaussian_find_peaks(xs_tmp, xopt[0], xopt[1], xopt[2], xopt[3], xopt[4], xopt[5])
    val = (a[0]-N*a[1]) / a[0]
    low_val, up_val = [np.min([0, val]), np.max([0,val])] 
    low_b, up_b =  [Bc, Bc] 
    while (not np.isclose(abs(val),0, atol = tol)):  
        estimates = (xopt[0],xopt[1],xopt[2],xopt[3],xopt[4],xopt[5])
        if(low_val * up_val >= 0):Bc = Bc - (np.sign(val) * np.sign(b[0] - b[1])*dB) #Finding extremes of Bisection
        else: Bc = (low_b + up_b) / 2 #Bisection
        lnz = llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, Bc)
        x, y = llr.calc_prob_distribution(final_df, Bc, lnz, xs_tmp)
        xopt, xcov = curve_fit(double_Gaussian, x, y, p0=estimates,)
        b,a = double_Gaussian_find_peaks(xs_tmp, xopt[0], xopt[1], xopt[2], xopt[3], xopt[4], xopt[5])
        val = (a[0]-N*a[1]) / a[0]
        if(low_val * val < 0): 
            up_val = val
            up_b = Bc
        if(up_val * val < 0): 
            low_val = val
            low_b = Bc
    DG = pd.DataFrame({'Bc':Bc,'lh':abs(b[1] - b[0]) * V,'dE':final_df['dE'][0], 'V':final_df['V'][0], 'Lt':final_df['Lt'][0], 'dP':a[0] - a[2], 'N':N, 'A1':xopt[0],'M1':xopt[1],'S1':xopt[2], 'A2':xopt[3],'M2':xopt[4],'S2':xopt[5], 'Emin':xs_tmp.min(), 'Emax':xs_tmp.max(), 'Epoints':len(xs_tmp)}, index = [0])
    return DG

def ReadDoubleGaussian(betas, dB, tol,N, final_df, fa_df,folder):
    DG_csv = folder + 'DG.csv'
    if os.path.isfile(DG_csv):
        DG = pd.read_csv(DG_csv)
    else:
        DG = find_critical_point(betas, dB, tol,N, final_df, fa_df,folder)
        DG.to_csv(DG_csv, index = False)
    return DG

def prepare_DG(LLR_folder, n_repeats, betas, dg_tol, dg_db):
    for nr in range(n_repeats):
        folder = f'{LLR_folder}{nr}/CSV/'
        obs_df = pd.read_csv(f'{folder}obs.csv')
        final_df = pd.read_csv(f'{folder}final.csv')
        ReadDoubleGaussian(betas,dg_db, dg_tol,1, final_df, obs_df, folder)

