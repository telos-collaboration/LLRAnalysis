import mpmath as mp
import numpy as np
import re
import pandas as pd
import os.path
import llranalysis.utils as utils
import llranalysis.standard as standard
import llranalysis.doubleGaussian as dg
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
import llranalysis.error as error
import llranalysis.Thermo as thermo

def ReadRep(file, new_rep, poly):
    txt_dS0 = "LLR Delta S "
    txt_swap = "New Rep Par S0 = "
    txt_S0 = "LLR S0 Central action"
    txt_a = "a_rho"
    txt_end = "Fixed a MC Step"
    txt_V = "Global size is"
    txt_Plaq = "Plaq a fixed"
    
    txt_fxda_E = "Fixed a MC Step"
    txt_fxda_py = "(Polyakov direction 0 =|FUND_POLYAKOV)"
    txt_fxda_S0 = "New LLR Param: "
    
    
    new_V = 1.
    new_Lt = 1.
    V = []
    Lt = []
    rep = []
    new_dS0 = 0.
    dS0 = []
    a = []
    S0 = []
    new_S0 = 0.
    swap_S0 = []
    rm_end = False
    rm_plaq = []
    new_plaq = 0.
    
    new_a = 0.
    new_E = 0.
    fa_a = []
    fa_S0 = []
    fa_E = []
    fa_py = []
    fa_py2 = []
    fa_rep = []
    fa_V = []
    fa_n = 0
    fa_N = []
    fa_Lt = []
    with open(file, "r") as a_file:
        
        for line in a_file:  
            if(not rm_end):
                #Check for starting action
                if(re.findall(txt_S0, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            new_S0 = float(word)
                elif(re.findall(txt_Plaq, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            new_plaq = float(word)
                elif(re.findall(txt_V, line) != []):
                    new_V = 1.
                    is_Lt = True
                    for word in re.split("x",re.split("\s",line)[-2]):
                        
                        if(utils.check_float(word)):
                            new_V *= float(word)
                            if is_Lt: 
                                new_Lt = float(word)
                                is_Lt = False
                        
                #Check for swap     
                elif(re.findall(txt_swap, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            if(new_S0 != float(word)):
                                new_S0 = float(word)
                                swap_S0.append(new_S0)
                            break
                elif(re.findall(txt_dS0, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            new_dS0 = float(word) 
                #Append a and S0        
                elif(re.findall(txt_a, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            a.append(float(word))
                            S0.append(new_S0)
                            dS0.append(new_dS0)
                            rep.append(new_rep)
                            V.append(new_V)
                            rm_plaq.append(new_plaq * 6. * new_V)
                            Lt.append(new_Lt)
                #End of RM
                elif(re.findall(txt_end, line) != []):
                    new_a = a[-1]
                    rm_end = True
                    for word in re.split("=", line):
                        if(utils.check_float(word)):
                            new_E = float(word)
                            if(not poly):
                                print("Shouldnt be here")
                                fa_S0.append(new_S0)
                                fa_a.append(new_a)
                                fa_E.append(new_E)
            else:
                if(re.findall(txt_fxda_E, line) != []):
                    for word in re.split("=", line):
                        if(utils.check_float(word)):
                            new_E = float(word)
                            if(not poly):
                                print("Shouldnt be here")
                                fa_S0.append(new_S0)
                                fa_a.append(new_a)
                                fa_E.append(new_E)
                elif((len(re.findall(txt_fxda_py, line)) > 1) and poly): 
                    py_tmp = 0.
                    for word in re.split("\s",re.split("=", line)[1]):
                        if(utils.check_float(word)):
                            py_tmp += float(word) ** 2
                    fa_py.append(py_tmp ** 0.5)
                    fa_py2.append(py_tmp)
                    fa_S0.append(new_S0)
                    fa_a.append(new_a)
                    fa_E.append(new_E)
                    fa_V.append(new_V)
                    fa_rep.append(new_rep)
                    fa_N.append(fa_n)
                    fa_n += 1
                    fa_Lt.append(new_Lt)
                elif(re.findall(txt_fxda_S0, line) != []):
                    bool_E = True
                    for word in re.split(r"[,\s]\s*", line):
                        if(utils.check_float(word) and bool_E):
                            new_S0 = float(word)
                            bool_E = False
                        elif(utils.check_float(word) and (not bool_E)):
                            new_a = float(word)
                            break;
    
    fa_S0 = np.array(fa_S0)
    fa_E = np.array(fa_E)
    S0 = np.array(S0) 
    dS0 = np.array(dS0) /  (2)
    a = - 1. * np.array(a) 
    fa_a = - 1. * np.array(fa_a)           
    if poly:
        fixed_a = pd.DataFrame(data = {'a':fa_a,'Ek':fa_S0,'S':fa_E, 'V':fa_V, 'Rep':fa_rep,'Poly':fa_py, 'PolySqr':fa_py2, 'n':fa_N})
    else:
        fixed_a = pd.DataFrame(data = {'a':fa_a,'Ek':fa_S0,'S':fa_E, 'V':fa_V, 'Rep':fa_rep, 'n':fa_N, 'Lt':fa_Lt})
    RM = pd.DataFrame(data = {'n':range(len(a)),'a':a,'Ek':S0,'dE':dS0, 'V':V, 'S': rm_plaq,'Rep':rep, 'Lt':Lt})
    final = pd.DataFrame(data = {'a':a[-1],'Ek':S0[-1],'dE':dS0[-1], 'V':V[-1], 'Lt':new_Lt}, index = [new_rep])
    return RM, fixed_a, final

def ReadFull(files, poly=True):
    fxa_df = pd.DataFrame()
    RM_df = pd.DataFrame()
    final_df = pd.DataFrame()
    for i, file in zip(range(len(files)),files):
        print(file)
        RM_tmp, fxa_tmp, final_tmp = ReadRep(file, i, poly=True)
        if final_df.empty:
            RM_df= RM_tmp.copy()
            fxa_df = fxa_tmp.copy()
            final_df = final_tmp.copy()
        else:       
            RM_df = RM_df.append(RM_tmp, ignore_index=True)
            fxa_df= fxa_df.append(fxa_tmp, ignore_index=True)
            final_df = final_df.append(final_tmp, ignore_index=True)
    RM_df = RM_df.sort_values(by=['Ek','n'], ignore_index = True)
    fxa_df = fxa_df.sort_values(by=['Ek','Rep'], ignore_index = True)
    final_df = final_df.sort_values(by=['Ek'], ignore_index = True)
    print(final_df)
    dE_new = (final_df['Ek'].values[1]-final_df['Ek'].values[0])
    final_df = final_df.assign(dE = np.ones(np.size(final_df['dE'].values)) * dE_new)
    final_df = final_df.assign(Ek = final_df['Ek'].values)
    RM_df = RM_df.assign(Ek = RM_df['Ek'].values)
    fxa_df = fxa_df.assign(Ek = fxa_df['Ek'].values)
    return fxa_df, RM_df, final_df

def SaveCSVFull(files,folder,poly=True):
    FA, RM, FNL = ReadFull(files, poly)
    RM.to_csv(folder + 'RM.csv', index = False)
    FA.to_csv(folder + 'fa.csv', index = False)
    FNL.to_csv(folder + 'final.csv', index = False)
    
def ReadCSVFull(folder):
    RM = pd.read_csv(folder + 'RM.csv')
    FA = pd.read_csv(folder + 'fa.csv')
    FNL = pd.read_csv(folder + 'final.csv')
    return RM, FA, FNL

def CSV(files,folder, poly=True):
    exists = os.path.isfile(folder + 'RM.csv') and os.path.isfile(folder + 'fa.csv') and os.path.isfile(folder + 'final.csv')
    if exists:
        print('Reading csv files')
        RM, FA, FNL = ReadCSVFull(folder)
    else:
        print('No CSV files creating them now')
        SaveCSVFull(files,folder,poly)
        RM, FA, FNL = ReadCSVFull(folder)
    return RM, FA, FNL

def ReadObservables(betas, final_df, fa_df, folder, file = 'obs.csv', calc_poly = True):
    print('Obs_DF')
    file_loc = folder + file
    exists = os.path.isfile(file_loc) 
    if exists:
        obs_DF = pd.read_csv(file_loc)
    else:
        obs_DF = calc_observables_full(betas,final_df,fa_df, calc_poly=calc_poly)
        obs_DF.to_csv(file_loc, index = False)
    obs_DF = obs_DF.sort_values(by=['b'], ignore_index=True)
    obs_DF.to_csv(file_loc, index = False)
    obs_DF = obs_DF[np.in1d(obs_DF['b'].values, betas)].reset_index()
    return obs_DF

def calc_observables_full(betas,final_df,fa_df, calc_poly = True):
    Eks = final_df['Ek'].values
    dE = (Eks[1] - Eks[0]) 
    V = final_df['V'].values[0]
    Lt = final_df['Lt'].values[0]
    aks = final_df['a'].values
    E = []
    E2 = []
    E4 = []
    poly = []
    poly_sq = []
    poly_4 = []
    for beta in betas: 
        print("Beta:",beta)
        E.append((calc_E(Eks, aks, beta)) / (6*V))
        E2.append((calc_E2(Eks, aks, beta)) / ((6*V) ** 2.))
        E4.append((calc_E4(Eks, aks, beta))/ ((6*V) ** 4.))
        if len(fa_df['S']) > 0 and calc_poly:
            lnz_obs = calc_fxa_lnZ(fa_df, final_df, beta)
            poly.append(calc_fxa_obs(fa_df, final_df, beta, 'Poly', 1., lnz_obs))
            poly_sq.append(calc_fxa_obs(fa_df, final_df, beta, 'Poly', 2.,lnz_obs))
            poly_4.append(calc_fxa_obs(fa_df, final_df, beta, 'Poly', 4.,lnz_obs))
        else:
            poly.append(0.)
            poly_sq.append(0.)
            poly_4.append(0.)
    SH =( np.array(E2) - (np.array(E)**2.)) * 6. * V 
    Xlp = (np.array(poly_sq) - (np.array(poly) ** 2.)) * (V/Lt)
    Blp = 0.
    binder = 1 - (np.array(E4) / (3 * (np.array(E2) ** 2.)))
    if(len(poly_sq) > 0): 
        if (min(poly_sq) != 0.): Blp = 1. - (np.array(poly_4) / (3*(np.array(poly_sq) ** 2.))) 
    return pd.DataFrame(data = {'b':betas,'u':E,'Cu':SH,'Bv':binder, 'lp':poly, 'Xlp':Xlp,'Blp':Blp, 'V':V * np.ones(len(poly)), 'Lt':Lt * np.ones(len(poly))})

def calc_lnZ(Eks, aks, beta, rmpf = False):
    pi_exp = 0.
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    Z = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta) 
        if not (A == mp.mpf(0.)):
            full_exp = mp.exp(pi_exp + (Ek*beta) + a*(dE/2.))
            T = A * dE / 2.
            shn = mp.sinh(T)
            Z += full_exp * shn / A
        else:
            print('A=0, beta =',beta,', a=', a)
            full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            Z += (dE / 2.) * full_exp
        pi_exp += a*dE
    ln_Z = mp.ln(2*Z)
    if not rmpf: ln_Z = float(ln_Z)
    return ln_Z

def calc_lnrho(final_df, E):
    pi_exp = 0.
    ln_rho = 0.
    dE = final_df['dE'][0]
    final_df = final_df.sort_values(by=['Ek'], ignore_index=True)
    for En, an in zip(final_df['Ek'].values - (final_df['dE'].values/2), final_df['a'].values):
        if(E >= En and E < (En + dE)):
            ln_rho = (an * (E - En)) + pi_exp
            break
        else:
            pi_exp += an*dE
    if ln_rho == 0:print("Energy not in range:",E)
    return ln_rho

def calc_E(Eks, aks, beta):
    pi_exp = 0.
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    E = mp.mpf(0)
    Z = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        if not (A == mp.mpf(0.)):
            full_exp = mp.exp(pi_exp + (Ek*beta) + a*(dE/2.))
            T = A * dE / 2.
            shn = mp.sinh(T)
            Z += full_exp * shn / A
            E += full_exp *( (Ek - (1./A))*shn + (dE/2.)*mp.cosh(T)) / A
        else:
            print('A=0, beta =',beta,', a=', a)
            full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            E += full_exp * (dE / 2.) * Ek
            Z += (dE / 2.) * full_exp
        pi_exp += a*dE
    return E/Z

def calc_E2(Eks, aks, beta):
    pi_exp = - calc_lnZ(Eks, aks, beta, rmpf = True)
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    E2 = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        if not (A == mp.mpf(0.)):
            full_exp = 2. * mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.))/ A
            T = A * dE / 2.
            E2 += full_exp *(mp.sinh(T)*((2./(A**2.)) - (2.*Ek/A) + ((Ek**2.) + ((dE/2.)**2.))) + mp.cosh(T)*((Ek*dE) - (dE/A))) 
        else:
            print('A=0, beta =',beta,', a=', a)
            full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            E2 += 2 * full_exp * (((Ek**2.) * dE/2.) + ((1./3.) * ((dE/2.) ** 3.)))
        pi_exp += a*dE
    #print(E / Z)
    return E2

def calc_E4(Eks, aks, beta):
    pi_exp = - calc_lnZ(Eks, aks, beta, rmpf = True)
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    E4 = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        T = A * dE / 2.
        if not (A == mp.mpf(0.)):
            full_exp = 2. * mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.) - 5.* mp.log(np.abs(A))) * 24. * (np.sign(A))
            M = (1.       - (Ek*A)            + ((Ek**2) * (A**2)/2) + (((dE/2)**2)* (A**2)/2) 
                              - ((Ek**3)* (A**3)/6) - (Ek* (A**3.) * ((dE/2)**2)/2)
                              + ((A**4)* (Ek**4)/24) + ((A**4)*(Ek**2)*((dE/2)**2)/4) + ((A**4)*((dE/2)**4)/24)) 
            N = ( - ((dE/2) * A) + ((A**2)*Ek*(dE/2)) 
                            - ((A**3)*(Ek ** 2)*(dE/2)/2) - ((A**3)*((dE/2)**3)/6) 
                            + ((A**4)*(Ek ** 3)*(dE/2)/6) + ((A**4)*((dE/2)**3)*Ek/6)) 
            E4_k = full_exp *(mp.sinh(T)*M + mp.cosh(T)*N)
            E4 += E4_k
            if np.abs(A) < 10**-7 : 
                print(mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.) + mp.log(mp.sinh(T))+ mp.log(abs(M))  + mp.log(mp.cosh(T))  - 5 * mp.log(np.abs(A))) * np.sign(A) * np.sign(M)
                      + mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.) + mp.log(abs(N)) + mp.log(mp.cosh(T))+  mp.log(mp.sinh(T))  - 5 * mp.log(np.abs(A))) * np.sign(A) * np.sign(N))
                print(E4_k)
                return np.NaN
                             #((Ek ** )*((dE/2)**2)/(2 * (A**2.))) 
        else:
            print('A=0, beta =',beta,', a=', a)
            #full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            #E2 += 2 * full_exp * (((Ek**2.) * dE/2.) + ((1./3.) * ((dE/2.) ** 3.)))
        pi_exp += a*dE
    #print(E / Z)
    return E4

def calc_EN(Eks, aks, beta, N):
    pi_exp = - calc_lnZ(Eks, aks, beta, rmpf = True)
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    EN = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        if not(A == mp.mpf(0.)):
            if np.abs(A) < 10**-7 : 
                    print('Very small A will get an error')
            full_exp =  np.math.factorial(N) * 2. * mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.))
            T = A * dE / 2.
            for n in range(N + 1):   
                snh_term = mp.mpf(0)
                for i in range(int(np.floor(n/2.)) + 1):
                    snh_term += ((dE / 2.)**(2*i)) * (Ek**(n-2*i)) / (np.math.factorial(2*i) * np.math.factorial(n - 2*i))
                ch_term = mp.mpf(0)
                if(n >0):
                    for i in range(1 ,int(np.ceil(n/2.)) + 1):
                        ch_term += ((dE / 2.)**(2*i - 1)) * (Ek**(n-2*i + 1)) / (np.math.factorial(2*i - 1) * np.math.factorial(n - 2*i + 1))
                EN += (snh_term*mp.sinh(T) + ch_term * mp.cosh(T)) * full_exp * ((-1.)**(N-n)) * (mp.power(A,(n - N - 1))) 
        else:
            print('A=0, beta =',beta,', a=', a)
        pi_exp += a*dE
    return EN

def calc_fxa_lnZ(fxa_df, final_df, beta):
    dE = final_df['dE'][0]
    final_df = final_df.sort_values(by=['Ek'], ignore_index=True)
    Ek = final_df['Ek'].values[0]; a = final_df['a'].values[0];
    S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values; ln_rhok = calc_lnrho(final_df, Ek); 
    VEV_exp = ((a+beta)*S - a*(Ek) + ln_rhok)  -np.log(len(S)) + np.log(dE)
    vmax = VEV_exp.max()
    Z = mp.mpf(0)   
    for Ek, a in zip(final_df['Ek'].values,final_df['a'].values):
        S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values 
        ln_rhok = calc_lnrho(final_df, Ek)
        VEV_exp = ((a+beta)*S - a*(Ek) + ln_rhok)  -np.log(len(S)) + np.log(dE) - vmax
        for ve in VEV_exp:    
            Z += mp.exp(mp.mpf(ve))
    ln_Z = (vmax + float(mp.ln(Z))) 
    return ln_Z

def calc_fxa_obs(fxa_df, final_df, beta, obs, n, lnz):
    dE = final_df['dE'][0]
    final_df = final_df.sort_values(by=['Ek'], ignore_index=True)
    B = 0.  
    for Ek, a in zip(final_df['Ek'].values,final_df['a'].values):
        S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values 
        P = fxa_df[fxa_df['Ek'].values == Ek][obs].values ** n
        ns =  fxa_df[fxa_df['Ek'].values == Ek]['n'].values 
        #plt.hist(S, histtype='step', bins = 100)
        ln_rhok = calc_lnrho(final_df, Ek)
        VEV_exp = (beta*S + a*(S - Ek) + ln_rhok - lnz)
        B += np.mean(P * dE * np.exp(VEV_exp))
        #plt.plot(Ek, np.mean(dE * np.exp(VEV_exp)) , 'kx')
    #plt.show()
    #print(obs,'^',n,':', B)
    return B

def calc_prob_distribution(final_df, beta, lnz, xs = np.array([])):
    if(xs.shape[0] == 0):
        xs = np.linspace(np.min(final_df['Ek'].values + final_df['dE'].values) / (6*final_df['V'].values[0]) , np.max(final_df['Ek'].values) / (6*final_df['V'].values[0]), 1000)
        #print('Here')
    ys = np.zeros(xs.size)
    for x, i in zip(xs,range(len(xs))):
        ys[i] = np.exp(calc_lnrho(final_df, x * (6*final_df['V'].values[0])) + beta*x*(6*final_df['V'].values[0]) - lnz)
    return np.array(xs), np.array(ys)

def prepare_data(LLR_folder, n_repeats, n_replicas, std_files, std_folder, betas, betas_critical, calc_poly = True):
    std_df, hist_df = standard.CSV(std_files, std_folder)
    for nr in range(n_repeats):
        files = [f'{LLR_folder}{nr}/Rep_{j}/out_0' for j in range(n_replicas)]
        RM, fa_df, final_df = CSV(files , f'{LLR_folder}{nr}/CSV/')
        comp_dF = ReadObservables(std_df['Beta'].values,final_df,fa_df,f'{LLR_folder}{nr}/CSV/',file = 'comparison.csv', calc_poly = calc_poly)
        full_dF = ReadObservables(betas,final_df,fa_df,f'{LLR_folder}{nr}/CSV/',file = 'obs.csv', calc_poly = calc_poly)
        critical_dF = ReadObservables(betas_critical,final_df,fa_df,f'{LLR_folder}{nr}/CSV/',file = 'obs_critical.csv', calc_poly=False)

def half_intervals(originalfolder, reducedfolder, mode='even'):
    for i in range(20):
        RM = pd.read_csv(originalfolder + str(i) + '/CSV/RM.csv')
        FA = pd.read_csv(originalfolder + str(i) + '/CSV/fa.csv')
        FNL = pd.read_csv(originalfolder + str(i) + '/CSV/final.csv')
        print(len(np.sort(np.unique(FNL['Ek'].values))[0:-1:2]))
        if mode=='even': Eks = (np.sort(np.unique(FNL['Ek'].values))[0:-1:2])
        elif mode=='odd': Eks = (np.sort(np.unique(FNL['Ek'].values))[1::2])
        elif mode=='mean':  Eks =( (np.sort(np.unique(FNL['Ek'].values))[0:-1:2]) + (np.sort(np.unique(FNL['Ek'].values))[1::2])) /2 
        RM_red = pd.DataFrame(); FA_red = pd.DataFrame(); FNL_red = pd.DataFrame();
        dE = Eks[1] - Eks[0]
        for ek in Eks:    
            RM_red =pd.concat([RM_red, RM[RM['Ek'] == ek]])
            FA_red =pd.concat([FA_red, FA[FA['Ek'] == ek]])
            FNL_red =pd.concat([FNL_red, FNL[FNL['Ek'] == ek]])
        RM_red['dE'] = np.ones_like(RM_red['Ek'])*dE
        FA_red['dE'] = np.ones_like(FA_red['Ek'])*dE
        FNL_red['dE'] = np.ones_like(FNL_red['Ek'])*dE
        RM_red.to_csv(reducedfolder + str(i) + '/CSV/RM.csv', index = False)
        FA_red.to_csv(reducedfolder+ str(i) + '/CSV/fa.csv', index = False)
        FNL_red.to_csv(reducedfolder + str(i) + '/CSV/final.csv', index = False)


def free_energy_df(boot_folder,n_repeats,num_samples,error_type):
    final_df = pd.read_csv(f'{boot_folder}/CSV/final.csv')
    V = final_df['V'][0]
    info_df = pd.read_csv(f'{boot_folder}/CSV/info.csv')
    Lt = info_df['Lt'][0]; Ls = info_df['Ls'][0];
    Ep = np.unique(final_df['Ek'])
    S,T,F,U = thermo.thermodynamics(boot_folder,n_repeats, Ep)
    Sigma = S.mean()
    F = (F + Sigma*T) / V
    mini,meta_mini, midi,meta_maxi, maxi = thermo.find_critical_region(6*V - U.mean(axis=0),T.mean(axis=0)) 
    ulim_ind = [max([maxi - 2,0]), min([mini + 2,len(Ep)-1])]
    ulim_int = [1-(U.mean(axis=0)[ulim_ind[0]]/ (6*V)),1-(U.mean(axis=0)[ulim_ind[1]]/ (6*V))]
    S_int,T_int,F_int,U_int = thermo.thermodynamics(boot_folder,n_repeats, np.linspace( (6*V)*ulim_int[0],  (6*V)*ulim_int[1], 1000))
    F_int = (F_int + Sigma*T_int) / V
    
    mini,meta_mini, midi,meta_maxi, maxi = thermo.find_critical_region(6*V - U_int.mean(axis=0),T_int.mean(axis=0))    


    meta_h = range(maxi,meta_mini+1)
    meta_c = range(meta_maxi,mini+1)
    unstable = range(meta_mini,meta_maxi+1)
    
    Tc = np.array([])
    P_min_F= np.array([])
    P_max_F= np.array([])
    up_min =  np.array([])
    up_max = np.array([])
    
    Tc = np.array([])
    P_min_F= np.array([])
    P_max_F= np.array([])
    up_min =  np.array([])
    up_max = np.array([])
    sigma = np.array([])

    xs = np.array([]);ys = np.array([])
    for i in range(n_repeats):
        try:
            f = interp1d(T_int[i,meta_h],F_int[i, meta_h])
            g = interp1d(T_int[i,meta_c],F_int[i, meta_c])
            t = np.linspace(max(min(T_int[i,meta_h]),min(T_int[i,meta_c])),min(max(T_int[i,meta_h]),max(T_int[i,meta_c])), 10000)
            tc_ind = np.where(T_int[i,:] == T_int[i,np.append(meta_h, meta_c)][np.argmin(abs(T_int[i,np.append(meta_h, meta_c)] - t[np.argmin(abs(f(t)-g(t)))]))])[0]
        except:
            print('Error no overlap in two meta-stable branches, likely misclassified : including unstable branch')
            mc = meta_c; mh = np.append(meta_h,unstable).flatten()
            f = interp1d(T_int[i,mh],F_int[i,mh])
            g = interp1d(T_int[i,mc],F_int[i, mc])
            t = np.linspace(max(min(T_int[i,mh]),min(T_int[i,mc])),min(max(T_int[i, mh]),max(T_int[i,mc])),  10000)
            tc_ind = np.where(T_int[i,:] == T_int[i,np.append(mh, mc)][np.argmin(abs(T_int[i,np.append(mh,mc)] - t[np.argmin(abs(f(t)-g(t)))]))])[0]
        tmin = T_int[i,tc_ind]
        pminf = F_int[i,tc_ind]
        pmaxf = F_int[i,unstable][np.argmin(abs(T_int[i,unstable] - tmin))]
        P_min_F = np.append(P_min_F,pmaxf)
        P_max_F = np.append(P_max_F ,pminf)
        Tc = np.append(Tc,tmin)
        
        up_min = np.append(up_min,1-(U_int[i,meta_h][np.argmin(abs(T_int[i,meta_h] - tmin))]/(6.*V)))
        up_max = np.append(up_max,1-(U_int[i,meta_c][np.argmin(abs(T_int[i,meta_c] - tmin))]/(6.*V)))
        
        beta = 1/tmin[0]
        final_df = pd.read_csv(f'{boot_folder}{i}/CSV/final.csv')
        lnz = float(calc_lnZ(final_df['Ek'].values, final_df['a'].values, beta))
        x, y = calc_prob_distribution(final_df, beta, lnz)
        xs = np.append(xs, x); ys = np.append(ys, y * (6*V))

    xs.shape = [n_repeats, len(x)]; ys.shape = [n_repeats, len(y)]

    F_int -= P_max_F.mean()
    F -= P_max_F.mean()
    P_min_F -= P_max_F.mean()
    P_max_F -= P_max_F.mean()
    
    sigma =  ( ( (Lt**3) * Ls * (P_min_F - P_max_F) ) / (2*Tc)) + (((Lt**2)*np.log(Ls) ) / (4*(Ls**2)) )
    
    for i in range(n_repeats):
        pd.DataFrame(data = {'Tc':Tc[i],'Fcmin':P_min_F[i],'Fcmax':P_max_F[i],
                                    'F':[list(F[i,:])], 'T':[list(T[i,:])], 'S':[list(S[i,:])],'U':[list(U[i,:])],
                                    'F_int':[list(F_int[i,:])], 'T_int':[list(T_int[i,:])], 'S_int':[list(S_int[i,:])],'U_int':[list(U_int[i,:])],
                                    'up-':up_min[i] ,'up+':up_max[i],
                                    'ind_c': [[mini,meta_mini, midi,meta_maxi, maxi]], 
                                    'E_Pb':[list(xs[i,:])], 'Pb':[list(ys[i,:])]}).to_csv(f'{boot_folder}{i}/CSV/F.csv')
    
    F_err = error.calculate_error_set(F,num_samples,error_type);
    T_err = error.calculate_error_set(T,num_samples,error_type);
    S_err = error.calculate_error_set(S,num_samples,error_type);
        
    F_int = F_int.mean(axis =0); T_int = T_int.mean(axis =0);
    S_int = S_int.mean(axis =0); U_int = U_int.mean(axis =0);
    F =F.mean(axis =0); T =T.mean(axis =0);
    S = S.mean(axis =0); U = U.mean(axis =0);
    
    Tc_err = error.calculate_error(Tc,num_samples,error_type); Tc = Tc.mean(axis=0);
    dF_err = error.calculate_error(P_min_F - P_max_F,num_samples,error_type); dF = (P_min_F - P_max_F).mean(axis=0);
    du_err = error.calculate_error(up_max -up_min,num_samples,error_type);du = (up_max -up_min).mean(axis=0);
    up_min_err = error.calculate_error(up_min,num_samples,error_type); up_min = up_min.mean(axis=0);
    up_max_err = error.calculate_error(up_max,num_samples,error_type); up_max = up_max.mean(axis=0);
    sigma_err = error.calculate_error(sigma,num_samples,error_type);sigma = sigma.mean();
    xs = xs.mean(axis = 0)
    ys_err = error.calculate_error_set(ys, num_samples, error_type)
    ys = ys.mean(axis = 0)

    pd.DataFrame(data = {'Tc':Tc,'Tc_err':Tc_err,
                         'dF':dF,'dF_err':dF_err,
                        'F':[list(F)], 'T':[list(T)], 'S':[list(S)],'U':[list(U)],
                        'F_err':[list(F_err)], 'T_err':[list(T_err)], 'S_err':[list(S_err)],
                        'F_int':[list(F_int)], 'T_int':[list(T_int)], 'S_int':[list(S_int)],'U_int':[list(U_int)],
                        'up-':up_min ,'up-_err':up_min_err ,'up+':up_max,'up+_err':up_max_err,
                         'du':du ,'du_err':du_err ,
                        'ind_c': [[mini,meta_mini, midi,meta_maxi, maxi]],
                        'E_Pb':[list(xs)], 'Pb':[list(ys)], 'Pb_err':[list(ys_err)],
                        'Sigma':sigma,'Sigma_err':sigma_err}).to_csv(f'{boot_folder}CSV/F.csv')  

def pre_dat(folder,V,up_min,up_max,N_intervals, betas, location):
    is_df = pd.read_csv(f'{folder}std.csv')
    x = np.array(betas)
    y = np.array([is_df[is_df['Beta']==b]['Plaq'] * 6 * V for b in betas]).flatten()
    fit = np.poly1d(np.polyfit(y,x,3))
    Eks = np.linspace(up_min,up_max, N_intervals)* 6 * V
    aks = fit(Eks)
    dE = (Eks[1]-Eks[0])*2
    output = ''
    for ek,ak in zip(Eks,aks):
        output+=f'{ek:.5f} {ak:.5f} {dE:.5f}\n'
    with open(location + 'pre.dat', 'w') as f:f.write(output)

def obs_boot(folder, n_repeat, num_samples, error_type):
    b = np.array([])
    u = np.array([]); Cu = np.array([]); Bv = np.array([]);
    u_err = np.array([]); Cu_err = np.array([]); Bv_err = np.array([]);
    lp = np.array([]); Xlp = np.array([]); 
    lp_err = np.array([]); Xlp_err = np.array([]); 
    for j in range(n_repeat):
        df = pd.read_csv(f'{folder}{j}/CSV/comparison.csv')
        u = np.append(u, df['u']);Cu = np.append(Cu, df['Cu']);Bv = np.append(Bv, df['Bv']); 
        lp = np.append(lp, df['lp']);Xlp = np.append(Xlp, df['Xlp']);
        b = np.append(b, df['b']) 
    len_b = len(df['b'])
    u.shape = [ n_repeat,len_b]; Cu.shape = [ n_repeat,len_b]; Bv.shape = [ n_repeat,len_b];
    lp.shape = [ n_repeat,len_b]; Xlp.shape = [ n_repeat,len_b]; 
    b.shape = [ n_repeat,len_b];
        
    u_err = error.calculate_error_set(u, num_samples, error_type);
    Cu_err = error.calculate_error_set(Cu, num_samples, error_type);
    Bv_err = error.calculate_error_set(Bv, num_samples, error_type);
    lp_err = error.calculate_error_set(lp, num_samples, error_type);
    Xlp_err = error.calculate_error_set(Xlp, num_samples, error_type);
        
    u = u.mean(axis=0);Cu = Cu.mean(axis=0);Bv = Bv.mean(axis=0);
    lp = lp.mean(axis=0);Xlp = Xlp.mean(axis=0);
    b = b.mean(axis=0);
    pd.DataFrame(data = {'b':b,'u':u,'Cu':Cu, 'Bv':Bv,
                               'u_err':u_err,'Cu_err':Cu_err, 'Bv_err':Bv_err,
                               'lp':lp,'Xlp':Xlp,
                                'lp_err': lp_err,'Xlp_err': Xlp_err}).to_csv(folder + 'CSV/comparison.csv')
    
    b = np.array([])
    bc_Xlp = np.array([]); Xlpc = np.array([]); 
    u = np.array([]); Cu = np.array([]); Bv = np.array([]);
    u_err = np.array([]); Cu_err = np.array([]); Bv_err = np.array([]);
    lp = np.array([]); Xlp = np.array([]); 
    lp_err = np.array([]); Xlp_err = np.array([]); 
    for j in range(n_repeat):
        df = pd.read_csv(f'{folder}{j}/CSV/obs.csv')
        u = np.append(u, df['u']);Cu = np.append(Cu, df['Cu']);Bv = np.append(Bv, df['Bv']); 
        lp = np.append(lp, df['lp']);Xlp = np.append(Xlp, df['Xlp']);
        b = np.append(b, df['b']) 
        y = [df.iloc[i]['Xlp'] for i in range(len(df)) if not np.isnan(df.iloc[i]['Xlp'])]
        x = [df.iloc[i]['b'] for i in range(len(df)) if not np.isnan(df.iloc[i]['Xlp'])]
        bc_Xlp = np.append(bc_Xlp, x[np.argmax(y)])
        Xlpc = np.append(Xlpc, max(y))
    len_b = len(df['b'])
    u.shape = [ n_repeat,len_b]; Cu.shape = [ n_repeat,len_b]; Bv.shape = [ n_repeat,len_b];
    lp.shape = [ n_repeat,len_b]; Xlp.shape = [ n_repeat,len_b]; 
    b.shape = [ n_repeat,len_b];
        
    u_err = error.calculate_error_set(u, num_samples, error_type);
    Cu_err = error.calculate_error_set(Cu, num_samples, error_type);
    Bv_err = error.calculate_error_set(Bv, num_samples, error_type);
    lp_err = error.calculate_error_set(lp, num_samples, error_type);
    Xlp_err = error.calculate_error_set(Xlp, num_samples, error_type);
        
    u = u.mean(axis=0);Cu = Cu.mean(axis=0);Bv = Bv.mean(axis=0);
    lp = lp.mean(axis=0);Xlp = Xlp.mean(axis=0);
    b = b.mean(axis=0);
    pd.DataFrame(data = {'b':b,'u':u,'Cu':Cu, 'Bv':Bv,
                               'u_err':u_err,'Cu_err':Cu_err, 'Bv_err':Bv_err,
                               'lp':lp,'Xlp':Xlp,
                                'lp_err': lp_err,'Xlp_err': Xlp_err}).to_csv(folder + 'CSV/obs.csv')
    
    b = np.array([])
    bc_Cu = np.array([]); Cuc = np.array([]); 
    bc_Bv = np.array([]); Bvc = np.array([]); 
    u = np.array([]); Cu = np.array([]); Bv = np.array([]);
    u_err = np.array([]); Cu_err = np.array([]); Bv_err = np.array([]);
    for j in range(n_repeat):
        df = pd.read_csv(f'{folder}{j}/CSV/obs_critical.csv')
        u = np.append(u, df['u']);Cu = np.append(Cu, df['Cu']);Bv = np.append(Bv, df['Bv']); 
        b = np.append(b, df['b']) 
        y = [df.iloc[i]['Cu'] for i in range(len(df)) if not np.isnan(df.iloc[i]['Cu'])]
        x = [df.iloc[i]['b'] for i in range(len(df)) if not np.isnan(df.iloc[i]['Cu'])]
        bc_Cu = np.append(bc_Cu,  x[np.argmax(y)])
        Cuc = np.append(Cuc, max(y))
        y = [df.iloc[i]['Bv'] for i in range(len(df)) if not np.isnan(df.iloc[i]['Bv'])]
        x = [df.iloc[i]['b'] for i in range(len(df)) if not np.isnan(df.iloc[i]['Bv'])]
        bc_Bv = np.append(bc_Bv, x[np.argmin(y)])
        Bvc = np.append(Bvc, min(y))

    len_b = len(df['b'])
    u.shape = [ n_repeat,len_b]; Cu.shape = [ n_repeat,len_b]; Bv.shape = [ n_repeat,len_b];
    b.shape = [ n_repeat,len_b];
        
    u_err = error.calculate_error_set(u, num_samples, error_type);
    Cu_err = error.calculate_error_set(Cu, num_samples, error_type);
    Bv_err = error.calculate_error_set(Bv, num_samples, error_type);
        
    u = u.mean(axis=0);Cu = Cu.mean(axis=0);Bv = Bv.mean(axis=0);
    lp = lp.mean(axis=0);Xlp = Xlp.mean(axis=0);
    b = b.mean(axis=0);
    pd.DataFrame(data = {'b':b,'u':u,'Cu':Cu, 'Bv':Bv,
                               'u_err':u_err,'Cu_err':Cu_err, 'Bv_err':Bv_err,}).to_csv(folder + 'CSV/obs_critical.csv')
    
    bc_Xlp_err = error.calculate_error(bc_Xlp, num_samples, error_type);
    bc_Cu_err = error.calculate_error(bc_Cu, num_samples, error_type);
    bc_Bv_err = error.calculate_error(bc_Bv, num_samples, error_type);
    Cuc_err = error.calculate_error(Cuc, num_samples, error_type);
    Bvc_err = error.calculate_error(Bvc, num_samples, error_type);
    Xlpc_err = error.calculate_error(Xlpc, num_samples, error_type);
    pd.DataFrame(data = {'b_Xlp':bc_Xlp,'b_Xlp_err':bc_Xlp_err,'Xlp':Xlpc, 'Xlp_err':Xlpc_err,
                         'b_Cu':bc_Cu,'b_Cu_err':bc_Cu_err,'Cu':Cuc, 'Cu_err':Cuc_err,
                         'b_Bv':bc_Bv,'b_Bv_err':bc_Bv_err,'Bv':Bvc, 'Bv_err':Bvc_err}, index=[0]).to_csv(folder + 'CSV/critical.csv')
    bc_Xlp = bc_Xlp.mean();
    bc_Cu = bc_Cu.mean();
    bc_Bv = bc_Bv.mean();
    Cuc = Cuc.mean();
    Bvc = Bvc.mean();
    Xlpc = Xlpc.mean();

    
def final_boot(folder, n_repeat, num_samples, error_type):
    ak = np.array([]);Ek = np.array([]); 
    for j in range(n_repeat):
        df = pd.read_csv(f'{folder}{j}/CSV/final.csv')
        ak = np.append(ak, df['a']);
    ak.shape = [n_repeat,len(df['a'])];
    Ek = df['Ek']
    dE = df['dE']
    ak_err = error.calculate_error_set(ak, num_samples, error_type);    
    ak = ak.mean(axis=0);
    pd.DataFrame(data = {'Ek':Ek,'a':ak,'a_err':ak_err, 'dE':dE,'V':df['V'],'Lt':df['Lt']}).to_csv(folder + 'CSV/final.csv', index = False)
    info_df = pd.read_csv(f'{folder}CSV/info.csv')
    info_df['dE'] = dE[0]
    info_df.to_csv(f'{folder}CSV/info.csv', index = False)
    
def DG_boot(folder, n_repeat, num_samples, error_type):
    bc = np.array([]);dup = np.array([]); dP= np.array([]);
    for j in range(n_repeat):
        df = pd.read_csv(f'{folder}{j}/CSV/DG.csv')
        bc = np.append(bc, df['Bc']);
        dup = np.append(dup, df['lh'] / (6*df['V']));
        dP =  np.append(dP, df['dP']);
    bc_err = error.calculate_error(bc, num_samples, error_type); 
    dup_err = error.calculate_error(dup, num_samples, error_type); 
    dP_err = error.calculate_error(dP, num_samples, error_type); 
    bc = bc.mean(axis=0);dup = dup.mean(axis=0);dP = dP.mean(axis=0);
    pd.DataFrame(data = {'bc':bc,'bc_err':bc_err,'dup':dup, 'dup_err':dup_err,'dP':dP, 'dP_err':dP_err}, index=[0]).to_csv(folder + 'CSV/DG.csv', index = False)

def prepare_all(folder):
    info_df = pd.read_csv(folder + 'CSV/info.csv')
    np.random.seed(info_df['seed'][0])
    seeds = np.random.randint(1000, size=3)
    if not os.path.isfile(folder + 'pre.dat'):
        print('Creating inital a and E')
        pre_dat(info_df['std_folder'][0],info_df['V'][0],info_df['umin'][0],info_df['umax'][0], info_df['n_replicas'][0], eval(info_df['IS_b'][0]), folder)
        return
    betas = np.linspace(eval(info_df['betas'][0])[0],eval(info_df['betas'][0])[1], eval(info_df['betas'][0])[2])
    betas_critical = np.linspace(eval(info_df['betas_critical'][0])[0],eval(info_df['betas_critical'][0])[1], eval(info_df['betas_critical'][0])[2])
    check_obs = os.path.isfile(folder + 'CSV/comparison.csv') * os.path.isfile(folder + 'CSV/obs.csv') * os.path.isfile(folder + 'CSV/obs_critical.csv')* os.path.isfile(folder + 'CSV/final.csv')* os.path.isfile(folder + 'CSV/critical.csv');
    check_dg = os.path.isfile(folder + 'CSV/DG.csv');
    check_free_energy = os.path.isfile(folder + 'CSV/F.csv'); 
    for i in range(info_df['n_repeats'][0]):
        check_obs *= os.path.isfile(folder + f'{i}/CSV/comparison.csv') * os.path.isfile(folder + f'{i}/CSV/obs.csv') * os.path.isfile(folder + f'{i}/CSV/obs_critical.csv')* os.path.isfile(folder + f'{i}/CSV/final.csv')
        check_dg *= os.path.isfile(folder + f'{i}/CSV/DG.csv')
        check_free_energy *= os.path.isfile(folder + f'{i}/CSV/F.csv')
    if not check_obs:
        np.random.seed(seeds[0])
        prepare_data(folder, info_df['n_repeats'][0],  info_df['n_replicas'][0], eval(info_df['std_files'][0]),  info_df['std_folder'][0], betas, betas_critical)
        final_boot(folder,info_df['n_repeats'][0] ,info_df['num_samples'][0],info_df['error_type'][0])
        obs_boot(folder,info_df['n_repeats'][0] ,info_df['num_samples'][0],info_df['error_type'][0])
    if not check_dg:
        np.random.seed(seeds[1])
        dg.prepare_DG(folder, info_df['n_repeats'][0], betas, info_df['dg_tol'][0], info_df['dg_db'][0]) 
        DG_boot(folder,info_df['n_repeats'][0] ,info_df['num_samples'][0],info_df['error_type'][0])
    if not check_free_energy:
        np.random.seed(seeds[2])
        free_energy_df(folder,info_df['n_repeats'][0],info_df['num_samples'][0],info_df['error_type'][0])