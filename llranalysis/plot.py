import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import llranalysis.llr as llr
import llranalysis.doubleGaussian as dg
import llranalysis.error as error
import llranalysis.standard as standard
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  
import matplotlib as mpl

def plot_RM_convergence(boot_folder, n_boots, interval, N_iterations):
    RM_df_names = [boot_folder + str(m) + '/CSV/RM.csv' for m in range(n_boots)]
    figure_folder = boot_folder + 'Figures/'
    RM = pd.read_csv(RM_df_names[0])
    Eks = np.unique(RM['Ek'])
    fig, ax =plt.subplots(figsize=(10,10)) # 
    lst_a = np.zeros((n_boots,N_iterations))
    for i, RM_df_name in enumerate(RM_df_names):
        RM = pd.read_csv(RM_df_name)
        Ek = Eks[interval]
        #plt.plot(-RM[RM['Ek'] == Ek]['a'].values) # axs[0]
        lst_a[i,:] = -RM[RM['Ek'] == Ek]['a'].values
    print(lst_a)
    plt.plot(lst_a.std(axis=0))
    plt.xlabel('RM iteration m') #axs[0].set_
    plt.ylabel('$\sigma_{a_n^{(m)}}$') #axs[0].set_
    print('Ek/6V:', Ek / (6*RM['V'].values[0]))
    print('DE/6V:', 2* RM['dE'].values[0] / (6*RM['V'].values[0]))
    plt.show()




def plot_RM_repeats(boot_folder, n_boots, interval):
    RM_df_names = [boot_folder + str(m) + '/CSV/RM.csv' for m in range(n_boots)]
    figure_folder = boot_folder + 'Figures/'
    RM = pd.read_csv(RM_df_names[0])
    Eks = np.unique(RM['Ek'])
    fig, ax =plt.subplots(figsize=(10,10)) # 
    lst_a = []
    for RM_df_name in RM_df_names:
        RM = pd.read_csv(RM_df_name)
        Ek = Eks[interval]
        plt.plot(-RM[RM['Ek'] == Ek]['a'].values) # axs[0]
        plt.xlabel('RM iteration m') #axs[0].set_
        plt.ylabel('$a_n^{(m)}$') #axs[0].set_
        lst_a.append(-RM[RM['Ek'] == Ek]['a'].values[-1])
    print('Ek/6V:', Ek / (6*RM['V'].values[0]))
    print('DE/6V:', 2* RM['dE'].values[0] / (6*RM['V'].values[0]))
    axsins = inset_axes(ax, width = 3, height = 3, loc='upper left',
                   bbox_to_anchor=(0.5,1-0.4,.3,.3), bbox_transform=ax.transAxes)
    axsins.hist(lst_a, orientation="horizontal",histtype='step')
    axsins.set_ylabel('$a_n$')
    axsins.set_xticks([])
    axsins.tick_params(axis='both', which='major', labelsize=15)
    axsins.tick_params(axis='both', which='minor', labelsize=15)
    axsins.locator_params(axis="y", nbins=5)
    plt.show()

def plot_RM_swaps(boot_folder, repeat, cmap):
    RM_df = pd.read_csv(boot_folder + str(repeat) + '/CSV/RM.csv')
    RM_df = RM_df.sort_values(by=['Rep','n'], ignore_index = True)
    print('DE/6V:', 2* RM_df['dE'].values[0] / (6*RM_df['V'].values[0]))
    cols = mpl.colormaps[cmap]
    max_N = 0
    for i, n in enumerate(np.unique(RM_df['Rep'])):
        an = -RM_df[RM_df['Rep'] == n]['a'].values
        plt.plot(np.arange(1,len(an)+1), an,lw=1, c= cols(i / len(np.unique(RM_df['Rep']))) )
        plt.xlabel('RM iteration $m$') # ,fontsize = 30
        plt.ylabel('$a_n^{(m)}$') # , fontsize = 30
        if max_N < len(an) + 1: max_N = len(an)+1 
    plt.ylim([-RM_df['a'].max(), -RM_df['a'].min()])
    plt.xlim(1, max_N)
    plt.show()

def plot_RM_swaps_fancy(boot_folder, repeat, cmap):
    plt.figure(figsize=(10,5))
    RM_df = pd.read_csv(boot_folder + str(repeat) + '/CSV/RM.csv')
    RM_df = RM_df.sort_values(by=['Rep','n'], ignore_index = True)
    print('DE/6V:', 2* RM_df['dE'].values[0] / (6*RM_df['V'].values[0]))
    cols = mpl.colormaps[cmap]
    max_N = 0
    for i, n in enumerate(np.unique(RM_df['Rep'])):
        an = -RM_df[RM_df['Rep'] == n]['a'].values
        plt.plot(np.arange(1,len(an)+1), an,lw=1, c= cols(i / len(np.unique(RM_df['Rep']))) )
        #plt.xlabel('RM iteration $m$') # ,fontsize = 30
        #plt.ylabel('$a_n^{(m)}$') # , fontsize = 30
        if max_N < len(an) + 1: max_N = len(an)+1 
    plt.ylim([-RM_df['a'].max(), -RM_df['a'].min()])
    plt.xlim(1, max_N)
    plt.axis('off')
    plt.show()

def plot_comparison_histograms(boot_folder, n_repeats, std_files, std_folder,num_samples = 200, error_type= 'standard deviation'):
    std_df, hist_df = standard.CSV(std_files, std_folder)
    colours = ['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
    for i, beta in enumerate(std_df['Beta'].values):
        print('beta:',beta)
        xs = np.array([]);ys = np.array([])
        for nr in range(n_repeats):
            final_df = pd.read_csv(f'{boot_folder}{nr}/CSV/final.csv')
            V = final_df['V'].values[0]
            lnz = float(llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, beta))
            x, y = llr.calc_prob_distribution(final_df, beta, lnz)
            xs = np.append(xs, x); ys = np.append(ys, y)
        xs.shape = [n_repeats, len(x)]; ys.shape = [n_repeats, len(y)]
        xs = xs.mean(axis = 0)
        ys_err = error.calculate_error_set(ys, num_samples, error_type)
        ys = ys.mean(axis = 0)
        #plt.plot(xs,(ys + ys_err)* (6*V), 'b--')
        plt.plot(xs,ys* (6*V), 'b-')
        #plt.plot(xs,(ys - ys_err)* (6*V), 'b--')
        hist_tmp = hist_df[hist_df['Beta'] == beta]['Hist'].values
        bins_tmp = hist_df[hist_df['Beta'] == beta]['Bins'].values
        plt.plot(bins_tmp,hist_tmp, c = 'darkorange', ls = '--') #, lw = 1
    print('DE/6V:', 2* final_df['dE'].values[0] / (6*final_df['V'].values[0]))
    plt.yticks([])
    plt.plot(np.NaN, np.NaN, 'b-', label='LLR')
    plt.plot(np.NaN, np.NaN, c='darkorange', ls='--', label='Importance sampling')
    plt.legend() 
    plt.ylabel('$P_{\\beta}(u_p)$' )
    plt.xlabel('$u_p$')
    plt.show()


def fxa_hist(boot_folder, selected_repeat):
    _, fxa_df, final_df = llr.ReadCSVFull(f'{boot_folder}{selected_repeat}/CSV/')
    V = final_df['V'].values[0]   
    for Ek, a in zip(final_df['Ek'].values,final_df['a'].values):
        S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values 
        S = S[S != 0]
    plt.hist(fxa_df['S'].values  / (6*V), histtype='step', bins = 100, density = True)
    plt.xlabel('$u_p$', fontsize = 30)
    plt.ylabel('$P(u_p)$', fontsize = 30)
    plt.yticks([],[])
    plt.show()

def plot_DG(LLR_folder, selected_repeat):
    DG = pd.read_csv(f'{LLR_folder}{selected_repeat}/CSV/DG.csv').iloc[0]
    final_df = pd.read_csv(f'{LLR_folder}{selected_repeat}/CSV/final.csv')
    Bc = DG['Bc']
    print('Bc: ', Bc)
    xopt = [DG['A1'],DG['M1'],DG['S1'], DG['A2'],DG['M2'],DG['S2']]
    V = 6*final_df['V'].values[0]
    lnz = llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, Bc)
    x, y = llr.calc_prob_distribution(final_df, Bc, lnz)
    plt.plot(x, dg.double_Gaussian(x, xopt[0], xopt[1], xopt[2], xopt[3], xopt[4], xopt[5]), 'm--', label = 'Fitted Double Gaussian',lw=1)
    plt.plot(x,y, 'b-', lw = 1,label= 'LLR')
    xs_tmp = np.linspace(DG['Emin'],DG['Emax'],int(DG['Epoints']))
    b,a = dg.double_Gaussian_find_peaks(xs_tmp, xopt[0], xopt[1], xopt[2], xopt[3], xopt[4], xopt[5])
    plt.arrow(b[1], a[0]/2, b[0] - b[1],0,width = 0.001*a[0], length_includes_head=True, color='k')
    plt.arrow(b[0], a[0]/2, b[1] - b[0],0,width = 0.001*a[0], length_includes_head=True, color='k')
    plt.text(b[1] - (b[1] - b[0])/2, (a[0]*1.06)/2, '$\\Delta \\langle u_p \\rangle_{\\beta_c}$',va ='baseline', horizontalalignment='center',fontsize=30)
    plt.yticks([])
    plt.legend(fontsize=20)
    plt.axvline(b[0]  , ls = '--', c= 'k', label='Peak locations')
    plt.axvline(b[1] , ls = '--', c= 'k')
    plt.xlabel('$ u_p $', fontsize=30)
    plt.ylabel('$P_{\\beta_c}( u_p )$',fontsize=30)
    N = 1
    plt.xlim([np.min(x[y > ((0.1 * (max(y) / N)))]),np.max(x[y > ((0.1 * (max(y) / N)))])])
    plt.legend(fontsize=20)
    plt.show()


def prepare_table(boot_folders):
    string_row = ''
    for bf in boot_folders:
        final_df = pd.read_csv(f'{bf}0/CSV/final.csv')
        V = final_df['V'].values[0]
        DE = final_df['dE'].values[0]*2 / (6*V)
        E_min = min(1 - (final_df['Ek'].values/ (6*V))) - DE/2 
        E_max = max(1 - (final_df['Ek'].values/ (6*V))) + DE/2
        N = len(final_df['Ek'].values)
        print()
        print(2*((E_max - E_min) / DE) - 1, N)
        string_row += f'${N}$ & ${DE:.4f}$ & ${E_min:.4f}$ & ${E_max:.4f}$ \\\\ \n'
    print(string_row)