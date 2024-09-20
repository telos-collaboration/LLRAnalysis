from matplotlib import colors as colours
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from llranalysis import llr
import llranalysis.error as error
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib as mpl


def find_critical_region(E,b):
    dE = abs(E[1] - E[0])
    #cr = [len(InterpolatedUnivariateSpline(E,b - b[i]).roots())>=2 for i in range(len(b))]
    cr = []
    #Changed to deal with a problem in a single ensemble :(
    for i in range(len(b)):
        roots = InterpolatedUnivariateSpline(E,b - b[i]).roots()
        if (len(roots>=2)): 
            cr.append((max(roots) - min(roots) > dE/10)) 
        else: cr.append(False)
    mini = min(np.nonzero(cr)[0]); maxi= max(np.nonzero(cr)[0])
    midb = (b[mini] + b[maxi]) / 2. 
    spl = InterpolatedUnivariateSpline(E,b - midb).roots()
    midi  = np.argmin(abs(E - np.median(spl)))
    meta_mini = np.argmax(b*(E > E[midi]))
    meta_maxi = np.argmin(b + b.max()*(E > E[midi]))
    [maxi,midi,mini] = np.sort(np.array([(mini,E[mini]), (midi,E[midi]),(maxi,E[maxi])],dtype=[('ind',int),('temp',float)]), order='temp')['ind']
    [meta_mini,meta_maxi] = np.sort(np.array([(meta_mini,E[meta_mini]), (meta_maxi,E[meta_maxi])],dtype=[('ind',int),('temp',float)]), order='temp')['ind']   
    return mini,meta_mini, midi,meta_maxi, maxi


def thermodynamics(boot_folder,n_repeats, Es):
    logrho0 = 0
    S = np.array([]); F = np.array([]); U = np.array([]); T = np.array([]); style = np.array([]); order = np.array([]);
    for i in range(n_repeats):
        final_df = pd.read_csv(f'{boot_folder}{i}/CSV/final.csv')
        V = final_df['V'][0]; dE = final_df['dE'][0]; Eps = np.unique(final_df['Ek']);
        betas = np.interp(Es, Eps, -final_df['a'].values);
        logrho = np.array([])
        for beta, Ep in zip(betas, Es):     
            logrho = np.append(logrho,llr.calc_lnrho(final_df, Ep))
        S= np.append(S,logrho + logrho0)
        T= np.append(T,1/betas)
        U = np.append(U,6*V - Es)
        F = np.append(F,(6*V - Es) - (1/betas)*(logrho + logrho0))
    lenx = len(Es); leny = n_repeats;
    S = S.reshape([leny,lenx]);T = T.reshape([leny, lenx]); F = F.reshape([leny, lenx]);U = U.reshape([leny, lenx]);
    return S,T,F,U


def free_energy(boot_folder,n_repeats,num_samples=200, error_type = 'standard deviation',cmap = 'rainbow'):
    markersize = 5
    final_df = pd.read_csv(f'{boot_folder}0/CSV/final.csv')
    V = final_df['V'][0]; dE = final_df['dE'][0]; 
    Ep = np.unique(final_df['Ek'])
    S,T,F,U = thermodynamics(boot_folder,n_repeats, np.linspace(Ep.min(), Ep.max(), 100))
    F /= V
    Sigma = S.mean() / V
    S_int = S.mean(axis=0); T_int = T.mean(axis=0); F_int = F.mean(axis=0); U_int = U.mean(axis=0);
    S_int_err = error.calculate_error_set(S,num_samples,error_type); T_int_err = error.calculate_error_set(T,num_samples,error_type);
    F_int_err = error.calculate_error_set(F,num_samples,error_type); U_int_err = error.calculate_error_set(U,num_samples,error_type);
    #S_int_err = S.std(axis=0, ddof=1); T_int_err = T.std(axis=0, ddof=1); F_int_err = F.std(axis=0, ddof=1); U_int_err = U.std(axis=0, ddof=1);
    F_red_int_avr = (F + Sigma*T).mean(axis=0);
    F_red_int_err = error.calculate_error_set((F + Sigma*T),num_samples,error_type);
    
    mini,meta_mini, midi,meta_maxi, maxi = find_critical_region(6*V - U_int,T_int)    
    
    S,T,F,U = thermodynamics(boot_folder,n_repeats, Ep)
    F /= V
    ak = (1/ T).mean(axis = 0)
    ak_err = error.calculate_error_set((1/ T),num_samples,error_type);
    S_avr = S.mean(axis=0); T_avr = T.mean(axis=0); F_avr = F.mean(axis=0); U_avr = U.mean(axis=0);
    S_err = error.calculate_error_set(S,num_samples,error_type); T_err = error.calculate_error_set(T,num_samples,error_type);
    F_err = error.calculate_error_set(F,num_samples,error_type); U_err = error.calculate_error_set(U,num_samples,error_type);
    #S_err = S.std(axis=0, ddof=1); T_err = T.std(axis=0, ddof=1); F_err = F.std(axis=0, ddof=1); U_err = U.std(axis=0, ddof=1);
    F_red_avr = (F + Sigma*T).mean(axis=0) 
    F_red_err = error.calculate_error_set((F + Sigma*T),num_samples,error_type);

    fig, ax = plt.subplots(figsize=(10,10))
    plt.plot(T_int[:maxi+1],F_red_int_avr[:maxi+1] ,'k-'); plt.plot(T_int[mini:],F_red_int_avr[mini:],'k-')
    plt.plot(T_int[maxi:meta_mini+1],F_red_int_avr[maxi:meta_mini+1],'b-'); plt.plot(T_int[meta_maxi:mini+1],F_red_int_avr[meta_maxi:mini+1],'b-');
    plt.plot(T_int[meta_mini:midi+1],F_red_int_avr[meta_mini:midi+1],'r-'); plt.plot(T_int[midi:meta_maxi+1],F_red_int_avr[midi:meta_maxi+1],'r-');
    Fmax = max(F_red_int_avr[meta_maxi],  F_red_int_avr[meta_mini])
    Fmin = 2*F_red_int_avr[mini]-Fmax
    inds = F_red_avr > Fmin
    norm = colours.Normalize(vmin=0, vmax=inds.sum() + 2, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='gist_heat')
    cols = cmap
    #cols = mpl.colormaps[cmap]
    #cs = np.array([(mapper.to_rgba(x)) for x in np.linspace(0,inds.sum(), num=inds.sum())])
    for c, (x, x_err, y, y_err) in enumerate(zip(T_avr[inds],T_err[inds], F_red_avr[inds], F_red_err[inds])):
        plt.errorbar(x,y ,xerr=x_err, yerr= y_err, color = cols(c / (len(T_avr[inds])-1)), marker = 'o', ms = markersize)
    plt.xlabel('t', fontsize=30)
    plt.ylabel('$f$', fontsize=30)
    Fmax = max(F_red_int_avr[meta_maxi],  F_red_int_avr[meta_mini])
    plt.ylim([Fmin,(2*Fmax-F_red_int_avr[maxi])])
    plt.xlim([T_int[mini+1],T_int[maxi - 3]])
    plt.locator_params(axis="x", nbins=5)
    axsins = inset_axes(ax, width = 3, height = 3, loc='upper left',
                   bbox_to_anchor=(0.55,1-0.3,.3,.3), bbox_transform=ax.transAxes)

    axsins.plot((6*V - U_int[:maxi+1])/(6*V), 1 / T_int[:maxi+1],'k-'); plt.plot((6*V -U_int[mini:])/(6*V),1 / T_int[mini:],'k-')
    axsins.plot((6*V -U_int[maxi:meta_mini+1])/(6*V), 1 / T_int[maxi:meta_mini+1],'b-'); plt.plot((6*V -U_int[meta_maxi:mini+1])/(6*V),1/T_int[meta_maxi:mini+1],'b-');
    axsins.plot((6*V - U_int[meta_mini:midi+1])/(6*V), 1 / T_int[meta_mini:midi+1],'r-'); plt.plot((6*V -U_int[midi:meta_maxi+1])/(6*V),1/T_int[midi:meta_maxi+1],'r-');
    Emax = max((( 6*V - U_int[meta_maxi]))/(6*V),  ((6*V - U_int[meta_mini])/(6*V)))
    axsins.set_ylim([1 / T_int[maxi - 1], 1 / T_int[mini+1]])
    axsins.set_xlim([ ((6*V - U_int[maxi - 2]))/(6*V),  ((6*V - U_int[mini+2]))/(6*V)])
    
    axsins.set_xlabel('$u_p$', fontsize=30)
    axsins.set_ylabel('$a_n$', fontsize=30)
    for c, (x, y, y_err) in enumerate(zip((6*V - U_avr[inds])/(6*V),ak[inds],ak_err[inds])):
        axsins.errorbar(x , y,y_err, color = cols(c/(len(ak[inds])-1)), marker = 'o',ms = markersize)
    plt.show()

def plot_ak_dist_potential(boot_folder, n_repeats, beta, ulim, blim,num_samples=200, error_type = 'standard deviation'):
    print(beta)
    xs = np.array([]);ys = np.array([]); a = np.array([]); Eks = np.array([]);
    for nr in range(n_repeats):
        final_df = pd.read_csv(f'{boot_folder}{nr}/CSV/final.csv')
        V = final_df['V'].values[0]
        lnz = float(llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, beta))
        x, y = llr.calc_prob_distribution(final_df, beta, lnz)
        xs = np.append(xs, x); ys = np.append(ys, y)
        a = np.append(a, final_df['a'].values); Eks = np.append(Eks, final_df['Ek'].values); 
    xs.shape = [n_repeats, len(x)]; ys.shape = [n_repeats, len(y)]; a.shape = [n_repeats, len(final_df['a'].values)]; Eks.shape = [n_repeats, len(final_df['Ek'].values)];
    xs = xs.mean(axis = 0); ys = ys.mean(axis = 0); 
    a_err = error.calculate_error_set(a,num_samples,error_type);
    a = a.mean(axis = 0); Eks = Eks.mean(axis = 0) #/ (6*V); 
    fig, axs = plt.subplots(3,1, figsize=(10,18), gridspec_kw={'height_ratios': [5, 5,5]})
    Emax = ulim[1]
    Emin = ulim[0]
    #lnz = llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, beta)
    #xs , ys = llr.prob_distribution(final_df, beta, lnz)
    spl = InterpolatedUnivariateSpline(Eks,a + beta).roots()
    axs[0].clear()
    axs[0].errorbar((Eks/ (6*V)), -a, a_err)
    axs[0].axhline(beta,ls='--',c='r')
    for s in (spl): axs[0].plot([ (s/ (6*V)), (s/ (6*V))], [beta,blim[0]],'m--')
    axs[0].set_ylim(blim[0],blim[1])
    axs[0].set_xlim(Emin,Emax)
    axs[0].set_ylabel('$a_n$', fontsize=30)
    #axs[0].text((106200/ (6*V)), beta, "$\\beta_{CV}$",horizontalalignment='left',verticalalignment='top',fontsize=30)
    axs[0].set_xticks([])
    axs[1].clear()
    axs[1].set_ylabel('$P_{\\beta}(u_p)$', fontsize=30)
    axs[1].set_xticks([])
    axs[1].set_yticklabels([])
    f = interp1d(xs ,ys)
    ymax = f(spl/(6*V))
    axs[1].fill_between(xs, 0, ys, alpha = 0.5)
    for s,ym in zip(spl,ymax): axs[1].plot([s/ (6*V),s/ (6*V)],[1,0],ls='--',c='m')
    axs[1].set_ylim(0,0.002)
    axs[1].set_xlim(Emin,Emax)
    axs[2].clear()
    axs[2].set_xlabel('$u_p$', fontsize=30)
    axs[2].set_ylabel('$W_{\\beta}(u_p)$', fontsize=30)
    f = interp1d(xs * 6 * V ,-np.log(ys))
    ymax = f(spl)
    for s,ym in zip(spl,ymax): axs[2].plot([s/ (6*V),s/ (6*V)],[100,ym],ls='--',c='m')
    axs[2].set_yticklabels([])
    #axs[2].set_xticklabels()
    axs[2].set_ylim(6.5,8.5)
    axs[2].set_xlim(Emin, Emax)
    plt.tight_layout()
    axs[2].plot(xs  ,-np.log(ys))
    plt.show()


def plot_fxa_polyakovloop_critical(boot_folder,n_repeats,selected_repeat, plt_all = True,num_samples=200, error_type = 'standard deviation'):
    final_df = pd.read_csv(f'{boot_folder}0/CSV/final.csv')
    V = final_df['V'][0]; dE = final_df['dE'][0]; 
    Ep = np.unique(final_df['Ek'])
    S,T,F,U = thermodynamics(boot_folder,n_repeats, np.linspace(Ep.min(), Ep.max(), 100))
    F /= V
    Sigma = S.mean() / V
    S_int = S.mean(axis=0); T_int = T.mean(axis=0); F_int = F.mean(axis=0); U_int = U.mean(axis=0);
    S_int_err = error.calculate_error_set(S,num_samples,error_type); T_int_err = error.calculate_error_set(T,num_samples,error_type);
    F_int_err = error.calculate_error_set(F,num_samples,error_type); U_int_err = error.calculate_error_set(U,num_samples,error_type);
    F_red_int_avr = (F + Sigma*T).mean(axis=0);
    F_red_int_err = error.calculate_error_set((F + Sigma*T),num_samples,error_type);
    
    mini,meta_mini, midi,meta_maxi, maxi = find_critical_region(6*V - U_int,T_int)  
    S,T,F,U = thermodynamics(boot_folder,n_repeats, Ep)
    F /= V
    ak = (1/ T).mean(axis = 0)
    ak_err = error.calculate_error_set((1/ T),num_samples,error_type);
    S_avr = S.mean(axis=0); T_avr = T.mean(axis=0); F_avr = F.mean(axis=0); U_avr = U.mean(axis=0);
    S_err = error.calculate_error_set(S,num_samples,error_type); T_err = error.calculate_error_set(T,num_samples,error_type);
    F_err = error.calculate_error_set(F,num_samples,error_type); U_err = error.calculate_error_set(U,num_samples,error_type);
    F_red_avr = (F + Sigma*T).mean(axis=0) 
    F_red_err = error.calculate_error_set((F + Sigma*T),num_samples,error_type);
    Fmax = max(F_red_int_avr[meta_maxi],  F_red_int_avr[meta_mini])
    Fmin = 2*F_red_int_avr[mini]-Fmax
    inds = F_red_avr > Fmin

    tach_inds = (U_avr > U_int[meta_maxi]) * (U_avr < U_int[meta_mini])
    stable_inds = ((U_avr >= U_int[mini]) * (U_avr <= U_int[meta_maxi])) + ((U_avr >= U_int[meta_mini]) *(U_avr <= U_int[maxi]))
    normal_inds = ((U_avr < U_int[mini]) + (U_avr > U_int[maxi]))
    print(normal_inds)
    _, fxa_df, final_df = llr.ReadCSVFull(f'{boot_folder}{selected_repeat}/CSV/')
    V = final_df['V'].values[0]  
    if plt_all: 
        for Ek in Ep[normal_inds]:
            temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
            temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
            plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='k')
    
        for Ek in Ep[tach_inds]:
            temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
            temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
            plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='r')
        for Ek in Ep[stable_inds]:
            temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
            temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
            plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='b')
    else:
        Ek = Ep[normal_inds][0]
        temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
        temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
        plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='k')
        Ek = Ep[normal_inds][-1]
        temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
        temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
        plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='k')

        #Ek = Ep[tach_inds][0]
        #temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
        #temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
        #plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='r')
        #Ek = Ep[tach_inds][-1]
        #temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
        #temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
        #plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='r')

        #Ek = Ep[stable_inds][0]
        #temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
        #temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
        #plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='b')
        #Ek = Ep[stable_inds][-1]
        #temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
        #temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
        #plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='b')
        for Ek in Ep[tach_inds]:
            temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
            temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
            plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='r')
        for Ek in Ep[stable_inds]:
            temp_fxa_df = fxa_df[fxa_df['Ek'].values == Ek]
            temp_fxa_df = temp_fxa_df[temp_fxa_df['S'].values != 0]
            plt.hist(temp_fxa_df['Poly'].values, histtype='step', bins = 100, density = True, color='b')


    plt.xlabel('$|l_p|$', fontsize = 30)
    plt.ylabel('$P(|l_p|)$', fontsize = 30)
    plt.yticks([],[])
    plt.show()


    plt.show()





def free_energy_difference(boot_folder,n_repeats,num_samples=200, error_type = 'standard deviation',cmap = 'rainbow'):
    markersize = 5
    final_df = pd.read_csv(f'{boot_folder}0/CSV/final.csv')
    V = final_df['V'][0]; dE = final_df['dE'][0]; 
    Ep = np.unique(final_df['Ek'])
    S,T,F,U = thermodynamics(boot_folder,n_repeats, np.linspace(Ep.min(), Ep.max(), 100))
    F /= V
    Sigma = S.mean() / V
    S_int = S.mean(axis=0); T_int = T.mean(axis=0); F_int = F.mean(axis=0); U_int = U.mean(axis=0);
    S_int_err = error.calculate_error_set(S,num_samples,error_type); T_int_err = error.calculate_error_set(T,num_samples,error_type);
    F_int_err = error.calculate_error_set(F,num_samples,error_type); U_int_err = error.calculate_error_set(U,num_samples,error_type);
    #S_int_err = S.std(axis=0, ddof=1); T_int_err = T.std(axis=0, ddof=1); F_int_err = F.std(axis=0, ddof=1); U_int_err = U.std(axis=0, ddof=1);
    F_red_int_avr = (F + Sigma*T).mean(axis=0);
    F_red_int_err = error.calculate_error_set((F + Sigma*T),num_samples,error_type);
    
    mini,meta_mini, midi,meta_maxi, maxi = find_critical_region(6*V - U_int,T_int)    
    
    tach_inds = (U_int >= U_int[meta_maxi]) * (U_int <= U_int[meta_mini])
    conf_inds = ((U_int >= U_int[mini]) * (U_int <= U_int[meta_maxi])) 
    deconf_inds = ((U_int >= U_int[meta_mini]) *(U_int <= U_int[maxi]))
    critical_inds = tach_inds + conf_inds + deconf_inds    

    T_cri = np.linspace(min(T_int[critical_inds]), max(T_int[critical_inds]), 20)
    F_unstable = np.interp(T_cri, T_int[tach_inds], F_red_int_avr[tach_inds]);
    F_conf = np.interp(T_cri, T_int[conf_inds][::-1], F_red_int_avr[conf_inds][::-1]);
    F_deconf = np.interp(T_cri, T_int[deconf_inds][::-1], F_red_int_avr[deconf_inds][::-1]);
    P_un_co = (F_conf - F_unstable)*V
    P_un_de = (F_deconf - F_unstable)*V

    fig, ax = plt.subplots(figsize=(10,10))
    #plt.plot(T_int[:maxi+1],F_red_int_avr[:maxi+1] ,'k-'); plt.plot(T_int[mini:],F_red_int_avr[mini:],'k-')
    plt.plot(T_cri,P_un_co,'b-');
    plt.plot(T_cri,P_un_de,'r-');
    ymax = max(P_un_co.max(),P_un_de.max())
    ymin = min(P_un_co.min(),P_un_de.min())

    plt.ylim([ymin,ymax])
    #plt.xlim([T_cri[mini+1],T_cri[maxi - 3]])
    plt.locator_params(axis="x", nbins=5)
    plt.show()