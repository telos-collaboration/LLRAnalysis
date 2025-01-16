import matplotlib.pyplot as plt
import pandas as pd
import os.path
import numpy as np
import llranalysis.llr as llr
import llranalysis.error as error
import llranalysis.standard as standard
import llranalysis.utils as utils


def compare_dE_plot_a(
    boot_folders,
    n_repeats,
    blim,
    ulim,
    num_samples=200,
    error_type="standard deviation",
):
    colours = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
    ]
    j = 0
    plt.figure(figsize=(10, 20))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    for bf, nr in zip(boot_folders, n_repeats):
        final_df = pd.read_csv(f"{bf}0/CSV/final.csv")
        Eks = final_df["Ek"].values
        V = final_df["V"].values[0]
        dE = final_df["dE"].values[0]
        aks = np.zeros((nr, len(Eks)))
        for i in range(nr):
            final_df = pd.read_csv(f"{bf}{i}/CSV/final.csv")
            aks[i, :] = -final_df["a"].values
        aks_mean = aks.mean(axis=0)
        aks_error = error.calculate_error_set(aks, num_samples, error_type)
        # ax1.errorbar(Eks/(6*V),aks_mean,aks_error, fmt = colours[j] + '-', label = '$a^4 \Delta_E / 6\\tilde{V}$' + f'= {2*dE / (6*V) :.4f}')
        # ax2.errorbar(Eks/(6*V),aks_mean,aks_error, fmt = colours[j] + '-', label = '$a^4 \Delta_E / 6\\tilde{V}$' + f'= {2*dE / (6*V) :.4f}')
        ax1.errorbar(
            Eks / (6 * V),
            aks_mean,
            aks_error,
            fmt=colours[j] + "-",
            label="$\Delta_{u_p}$" + f"= {2 * dE / (6 * V):.4f}",
        )
        ax2.errorbar(
            Eks / (6 * V),
            aks_mean,
            aks_error,
            fmt=colours[j] + "-",
            label="$\Delta_{u_p}$" + f"= {2 * dE / (6 * V):.4f}",
        )
        j += 1
    ax1.set_ylabel("$a_n$")
    ax2.set_ylabel("$a_n$")
    ax1.locator_params(axis="x", nbins=7)
    ax2.locator_params(axis="x", nbins=7)
    ax1.legend()
    ax2.set_xlim(ulim)
    ax2.set_ylim(blim)
    ax2.set_xlabel("$u_p$")
    plt.subplots_adjust(hspace=0.05)
    plt.show()


def dE_DG_critical_beta(
    full_folders,
    reduced_folders,
    additional_folders,
    num_repeats,
    num_samples=200,
    error_type="standard deviation",
    plt_a=False,
):
    markersize = 10
    fig = plt.figure()
    miny = 100.0
    maxy = 0.0
    DG_bc_dE = np.array([])
    DG_bc_dE_err = np.array([])
    dEsq = np.array([])
    for i, rf in enumerate(additional_folders):
        c = "g"
        dEsq = np.append(
            dEsq,
            (
                (
                    2
                    * pd.read_csv(rf + "0/CSV/" + "final.csv")["dE"].values[0]
                    / (6.0 * pd.read_csv(rf + "0/CSV/" + "final.csv")["V"].values[0])
                )
                ** 2.0
            ),
        )
        DG_df = pd.DataFrame()
        for nr in range(num_repeats):
            DG_df = pd.concat([DG_df, pd.read_csv(f"{rf}{nr}/CSV/DG.csv")])
        plt.plot(dEsq[-1] * np.ones_like(DG_df["Bc"]), DG_df["Bc"], "kx")
        DG_bc_dE = np.append(DG_bc_dE, DG_df["Bc"].values.mean())
        DG_bc_dE_err = np.append(
            DG_bc_dE_err,
            error.calculate_error(
                DG_df["Bc"].values, num_samples, error_type=error_type
            ),
        )
        print(
            f"All intervals & {dEsq[-1] ** 0.5} & {DG_bc_dE[-1]} ({DG_bc_dE_err[-1]})"
        )
        if rf not in full_folders:
            plt.errorbar(
                dEsq[-1],
                DG_bc_dE[-1],
                DG_bc_dE_err[-1],
                color="darkorange",
                marker="o",
                capsize=10,
                capthick=5,
                ms=markersize,
            )
    DG_bc_0, DG_bc_0_err = utils.plot_extrap(
        dEsq, DG_bc_dE, DG_bc_dE_err, np.arange(len(dEsq)), c, max(dEsq)
    )
    plt.errorbar(
        0,
        DG_bc_0,
        DG_bc_0_err,
        color=c,
        marker="^",
        capsize=10,
        capthick=5,
        ms=markersize,
    )
    print(f"All intervals reduced & 0 & {DG_bc_0} ({DG_bc_0_err})")
    mindEsq = (np.sort(dEsq)[-1] + np.sort(dEsq)[-2]) / 2
    DG_bc_dE = np.array([])
    DG_bc_dE_err = np.array([])
    dEsq = np.array([])
    for i, rf in enumerate(reduced_folders):
        c = "b"
        dEsq = np.append(
            dEsq,
            (
                (
                    2
                    * pd.read_csv(rf + "0/CSV/" + "final.csv")["dE"].values[0]
                    / (
                        2
                        * 6.0
                        * pd.read_csv(rf + "0/CSV/" + "final.csv")["V"].values[0]
                    )
                )
                ** 2.0
            ),
        )
        DG_df = pd.DataFrame()
        for nr in range(num_repeats):
            DG_df = pd.concat([DG_df, pd.read_csv(f"{rf}{nr}/CSV/DG.csv")])
        DG_bc_dE = np.append(DG_bc_dE, DG_df["Bc"].values.mean())
        DG_bc_dE_err = np.append(
            DG_bc_dE_err,
            error.calculate_error(DG_df["Bc"].values, num_samples, error_type),
        )
        plt.errorbar(
            dEsq,
            DG_bc_dE,
            DG_bc_dE_err,
            color=c,
            marker="o",
            capsize=12,
            capthick=5,
            ms=markersize,
        )
        print(
            f"Even intervals & {dEsq[-1] ** 0.5} & {DG_bc_dE[-1]} ({DG_bc_dE_err[-1]})"
        )
    DG_bc_0, DG_bc_0_err = utils.plot_extrap(
        dEsq, DG_bc_dE, DG_bc_dE_err, np.arange(len(dEsq)), c, mindEsq
    )
    plt.errorbar(
        0,
        DG_bc_0,
        DG_bc_0_err,
        color=c,
        marker="^",
        capsize=12,
        capthick=5,
        ms=markersize,
    )
    print(f"Even intervals reduced & 0 & {DG_bc_0} ({DG_bc_0_err})")
    DG_bc_dE = np.array([])
    DG_bc_dE_err = np.array([])
    dEsq = np.array([])
    for i, rf in enumerate(full_folders):
        c = "darkorange"
        dEsq = np.append(
            dEsq,
            (
                (
                    2
                    * pd.read_csv(rf + "0/CSV/" + "final.csv")["dE"].values[0]
                    / (6.0 * pd.read_csv(rf + "0/CSV/" + "final.csv")["V"].values[0])
                )
                ** 2.0
            ),
        )
        if plt_a:
            for counter in range(num_repeats):
                micro_a = -pd.read_csv(f"{rf}{counter}/CSV/final.csv")["a"].values
                plt.plot(dEsq[-1] * np.ones_like(micro_a), micro_a, "m.")
        DG_df = pd.DataFrame()
        for nr in range(num_repeats):
            DG_df = pd.concat([DG_df, pd.read_csv(f"{rf}{nr}/CSV/DG.csv")])
        plt.plot(dEsq[-1] * np.ones_like(DG_df["Bc"]), DG_df["Bc"], "kx")
        if miny > min(DG_df["Bc"]):
            miny = min(DG_df["Bc"])
        if maxy < max(DG_df["Bc"]):
            maxy = max(DG_df["Bc"])
        DG_bc_dE = np.append(DG_bc_dE, DG_df["Bc"].values.mean())
        DG_bc_dE_err = np.append(
            DG_bc_dE_err,
            error.calculate_error(
                DG_df["Bc"].values, num_samples, error_type=error_type
            ),
        )
        plt.errorbar(
            dEsq[-1],
            DG_bc_dE[-1],
            DG_bc_dE_err[-1],
            color=c,
            marker="o",
            capsize=8,
            capthick=5,
            ms=markersize,
        )
    DG_bc_0, DG_bc_0_err = utils.plot_extrap(
        dEsq, DG_bc_dE, DG_bc_dE_err, np.arange(len(dEsq)), c, mindEsq
    )
    print(f"All intervals all points & 0 & {DG_bc_0} ({DG_bc_0_err})")
    plt.errorbar(
        0,
        DG_bc_0,
        DG_bc_0_err,
        color=c,
        marker="^",
        capsize=8,
        capthick=5,
        ms=markersize,
    )
    plt.ylabel("$\\beta_{CV}$")
    # plt.xlabel('$(a^4 \\Delta_E / (6\\tilde{V}))^2$')
    plt.xlabel("$(\\Delta_{u_p})^2$")
    plt.ylim([miny, maxy])
    # plt.errorbar(np.NaN, np.NaN,np.NaN,marker = 'o', color = 'g',label = 'All intervals, all points')
    plt.errorbar(
        np.NaN, np.NaN, np.NaN, marker="o", color="darkorange", label="All intervals"
    )
    plt.errorbar(np.NaN, np.NaN, np.NaN, marker="o", color="b", label="Odd intervals")
    plt.legend()
    plt.show()


def dE_DG_critical_plaq(
    full_folders,
    reduced_folders,
    additional_folders,
    num_repeats,
    num_samples=1000,
    error_type="standard deviation",
):
    markersize = 10
    fig = plt.figure()
    dEsq = np.array([])
    up_bc_dE = np.array([])
    up_bc_dE_err = np.array([])
    for i, rf in enumerate(additional_folders):
        c = "g"
        dEsq = np.append(
            dEsq,
            (
                2
                * pd.read_csv(rf + "0/CSV/" + "final.csv")["dE"].values[0]
                / (6.0 * pd.read_csv(rf + "0/CSV/" + "final.csv")["V"].values[0])
            )
            ** 2.0,
        )
        DG_df = pd.DataFrame()
        for nr in range(num_repeats):
            DG_df = pd.concat([DG_df, pd.read_csv(f"{rf}{nr}/CSV/DG.csv")])
        V = DG_df["V"].values[0]
        up_bc_dE = np.append(up_bc_dE, DG_df["lh"].values.mean() / (6 * V))
        up_bc_dE_err = np.append(
            up_bc_dE_err,
            error.calculate_error(
                DG_df["lh"].values / (6 * V), num_samples, error_type=error_type
            ),
        )
        print(
            f"All intervals & {dEsq[-1] ** 0.5} & {up_bc_dE[-1]} ({up_bc_dE_err[-1]})"
        )
        if rf not in full_folders:
            plt.errorbar(
                dEsq[-1],
                up_bc_dE[-1],
                up_bc_dE_err[-1],
                color="darkorange",
                marker="o",
                capsize=10,
                capthick=5,
                ms=markersize,
            )
    up_bc_0, up_bc_0_err = utils.plot_extrap(
        dEsq, up_bc_dE, up_bc_dE_err, np.arange(len(dEsq)), c, max(dEsq)
    )
    plt.errorbar(
        0,
        up_bc_0,
        up_bc_0_err,
        color=c,
        marker="^",
        capsize=10,
        capthick=5,
        ms=markersize,
    )
    print(f"All intervals reduced & 0 & {up_bc_0} ({up_bc_0_err})")
    mindEsq = (np.sort(dEsq)[-1] + np.sort(dEsq)[-2]) / 2
    up_bc_dE = np.array([])
    up_bc_dE_err = np.array([])
    dEsq = np.array([])
    for i, rf in enumerate(reduced_folders):
        c = "b"
        dEsq = np.append(
            dEsq,
            (
                (
                    2
                    * pd.read_csv(rf + "0/CSV/" + "final.csv")["dE"].values[0]
                    / (
                        2
                        * 6.0
                        * pd.read_csv(rf + "0/CSV/" + "final.csv")["V"].values[0]
                    )
                )
                ** 2.0
            ),
        )
        DG_df = pd.DataFrame()
        for nr in range(num_repeats):
            DG_df = pd.concat([DG_df, pd.read_csv(f"{rf}{nr}/CSV/DG.csv")])
        up_bc_dE = np.append(up_bc_dE, DG_df["lh"].values.mean() / (6 * V))
        up_bc_dE_err = np.append(
            up_bc_dE_err,
            error.calculate_error(
                DG_df["lh"].values / (6 * V), num_samples, error_type
            ),
        )
        plt.errorbar(
            dEsq[-1],
            up_bc_dE[-1],
            up_bc_dE_err[-1],
            color=c,
            marker="o",
            capsize=12,
            capthick=5,
            ms=markersize,
        )
        print(
            f"Even intervals & {dEsq[-1] ** 0.5} & {up_bc_dE[-1]} ({up_bc_dE_err[-1]})"
        )
    up_bc_0, up_bc_0_err = utils.plot_extrap(
        dEsq, up_bc_dE, up_bc_dE_err, np.arange(len(dEsq)), c, mindEsq
    )
    plt.errorbar(
        0,
        up_bc_0,
        up_bc_0_err,
        color=c,
        marker="^",
        capsize=12,
        capthick=5,
        ms=markersize,
    )
    print(f"Even intervals reduced & 0 & {up_bc_0} ({up_bc_0_err})")
    up_bc_dE = np.array([])
    up_bc_dE_err = np.array([])
    dEsq = np.array([])
    for i, rf in enumerate(full_folders):
        c = "darkorange"
        dEsq = np.append(
            dEsq,
            (
                (
                    2
                    * pd.read_csv(rf + "0/CSV/" + "final.csv")["dE"].values[0]
                    / (6.0 * pd.read_csv(rf + "0/CSV/" + "final.csv")["V"].values[0])
                )
                ** 2.0
            ),
        )
        DG_df = pd.DataFrame()
        for nr in range(num_repeats):
            DG_df = pd.concat([DG_df, pd.read_csv(f"{rf}{nr}/CSV/DG.csv")])
        up_bc_dE = np.append(up_bc_dE, DG_df["lh"].values.mean() / (6 * V))
        up_bc_dE_err = np.append(
            up_bc_dE_err,
            error.calculate_error(
                DG_df["lh"].values / (6 * V), num_samples, error_type=error_type
            ),
        )
        plt.errorbar(
            dEsq[-1],
            up_bc_dE[-1],
            up_bc_dE_err[-1],
            color=c,
            marker="o",
            capsize=8,
            capthick=5,
            ms=markersize,
        )
    up_bc_0, up_bc_0_err = utils.plot_extrap(
        dEsq, up_bc_dE, up_bc_dE_err, np.arange(len(dEsq)), c, mindEsq
    )
    print(f"All intervals all points & 0 & {up_bc_0} ({up_bc_0_err})")
    plt.errorbar(
        0,
        up_bc_0,
        up_bc_0_err,
        color=c,
        marker="^",
        capsize=8,
        capthick=5,
        ms=markersize,
    )
    plt.ylabel("$\Delta \\langle u_p \\rangle_{\\beta_{CV}}$")
    # plt.xlabel('$(a^4 \\Delta_E / (6\\tilde{V}))^2$')
    plt.xlabel("$(\\Delta_{u_p})^2$")
    # plt.errorbar(np.NaN, np.NaN,np.NaN,marker = 'o', color = 'g',label = 'All intervals, all points')
    plt.errorbar(
        np.NaN, np.NaN, np.NaN, marker="o", color="darkorange", label="All intervals"
    )
    plt.errorbar(np.NaN, np.NaN, np.NaN, marker="o", color="b", label="Odd intervals")
    plt.legend()
    plt.show()


def compare_dE_plot_y(
    boot_folders,
    n_repeats,
    std_files,
    std_folder,
    std_key,
    llr_key,
    label,
    num_samples=200,
    error_type="standard deviation",
    extrema="",
):
    colours = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
    ]
    std_df, hist_df = standard.CSV(std_files, std_folder)
    std_bs = std_df["Beta"].values
    std_ys = std_df[std_key].values
    std_ys_err = std_df[std_key + "_err"].values
    for i, (bf, nr) in enumerate(zip(boot_folders, n_repeats)):
        llr_comp_y = np.array([])
        llr_comp_b = np.array([])
        llr_full_y = np.array([])
        llr_full_b = np.array([])
        if extrema == "max" or extrema == "min":
            bmax = np.array([])
        for j in range(nr):
            comp_df = pd.read_csv(f"{bf}{j}/CSV/comparison.csv")
            llr_comp_y = np.append(llr_comp_y, comp_df[llr_key])
            llr_comp_b = np.append(llr_comp_b, comp_df["b"])
            full_df = pd.read_csv(f"{bf}{j}/CSV/obs_critical.csv")
            if full_df[llr_key].values.sum() == 0:
                full_df = pd.read_csv(f"{bf}{j}/CSV/obs.csv")
            llr_full_y = np.append(llr_full_y, full_df[llr_key])
            llr_full_b = np.append(llr_full_b, full_df["b"])
            if extrema == "max":
                bmax = np.append(bmax, full_df["b"].values[full_df[llr_key].argmax()])
            elif extrema == "min":
                bmax = np.append(bmax, full_df["b"].values[full_df[llr_key].argmin()])
        V = comp_df["V"].values[0]
        dE = pd.read_csv(f"{bf}0/CSV/final.csv")["dE"].values[0]
        llr_comp_b.shape = [nr, len(comp_df["b"])]
        llr_comp_y.shape = [nr, len(comp_df["b"])]
        llr_comp_b = llr_comp_b.mean(axis=0)
        llr_comp_y_err = error.calculate_error_set(llr_comp_y, num_samples, error_type)
        llr_comp_y = llr_comp_y.mean(axis=0)

        llr_full_b.shape = [nr, len(full_df["b"])]
        llr_full_y.shape = [nr, len(full_df["b"])]
        llr_full_b = llr_full_b.mean(axis=0)
        llr_full_y_err = error.calculate_error_set(llr_full_y, num_samples, error_type)
        if extrema == "max" or extrema == "min":
            bmax_avr = bmax.mean()
            bmax_err = error.calculate_error(bmax, num_samples, error_type)
            print(
                f"a^4 \Delta_E/ 6V= {2 * dE / (6 * V):.4f} beta_C = {bmax_avr} +/- {bmax_err}"
            )
            plt.axvspan(
                bmax_avr - bmax_err, bmax_avr + bmax_err, alpha=0.2, color=colours[i]
            )
            plt.axvline(bmax_avr, color=colours[i], linestyle="dashed")
        llr_full_y = llr_full_y.mean(axis=0)
        # plt.plot(llr_full_b, llr_full_y, colours[i] + '-', label = 'LLR  $a^4 \Delta_E/ 6\\tilde{V}$' + f'= {2*dE / (6*V):.4f}' ,zorder=i) #
        plt.plot(
            llr_full_b,
            llr_full_y,
            colours[i] + "-",
            label="LLR  $\Delta_{u_p}$" + f"= {2 * dE / (6 * V):.4f}",
            zorder=i,
        )  #
        plt.plot(llr_full_b, llr_full_y + llr_full_y_err, colours[i] + "--")
        plt.plot(llr_full_b, llr_full_y - llr_full_y_err, colours[i] + "--")
        plt.errorbar(
            llr_comp_b, llr_comp_y, llr_comp_y_err, fmt=colours[i] + "o", capsize=10
        )
    plt.errorbar(
        std_bs, std_ys, std_ys_err, fmt="k^", label="Importance sampling", capsize=10
    )
    plt.ylabel(label, fontsize=30)
    plt.xlabel("$\\beta$", fontsize=30)
    plt.legend()
    plt.locator_params(axis="x", nbins=7)
    plt.show()


def compare_dE_plot_y_difference(
    boot_folders,
    n_repeats,
    std_files,
    std_folder,
    std_key,
    llr_key,
    label,
    num_samples=200,
    error_type="standard deviation",
):
    colours = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
    ]
    std_df, hist_df = standard.CSV(std_files, std_folder)
    std_bs = std_df["Beta"].values
    std_ys = std_df[std_key].values
    std_ys_err = std_df[std_key + "_err"].values
    for i, (bf, nr) in enumerate(zip(boot_folders, n_repeats)):
        llr_comp_y = np.array([])
        llr_comp_b = np.array([])
        llr_full_y = np.array([])
        llr_full_b = np.array([])
        for j in range(nr):
            comp_df = pd.read_csv(f"{bf}{j}/CSV/comparison.csv")
            llr_comp_y = np.append(llr_comp_y, comp_df[llr_key])
            llr_comp_b = np.append(llr_comp_b, comp_df["b"])
            full_df = pd.read_csv(f"{bf}{j}/CSV/obs.csv")
            llr_full_y = np.append(llr_full_y, full_df[llr_key])
            llr_full_b = np.append(llr_full_b, full_df["b"])
        V = comp_df["V"].values[0]
        dE = pd.read_csv(f"{bf}0/CSV/final.csv")["dE"].values[0]
        llr_comp_b.shape = [nr, len(comp_df["b"])]
        llr_comp_y.shape = [nr, len(comp_df["b"])]
        llr_comp_b = llr_comp_b.mean(axis=0)
        a = llr_comp_y.mean(axis=0)
        da = error.calculate_error_set(llr_comp_y, num_samples, error_type)
        b = std_ys
        db = std_ys_err
        c = (a - b) / b
        dc = (a / b) * ((((da / a) ** 2) + ((db / b) ** 2)) ** 0.5)
        # plt.errorbar(llr_comp_b, c,dc, fmt = colours[i] + 'o', capsize=10, label = 'LLR  $a^4 \Delta_E/ 6\\tilde{V}$' + f'= {2*dE / (6*V):.4f}',zorder=i)
        plt.errorbar(
            llr_comp_b,
            c,
            dc,
            fmt=colours[i] + "o",
            capsize=10,
            label="LLR  $\Delta_{u_p}$" + f"= {2 * dE / (6 * V):.4f}",
            zorder=i,
        )
        print(llr_comp_b, dE / (3 * V))
        print(c, dc)
    # plt.errorbar(std_bs, 0.*std_ys, std_ys_err, fmt ='k^', label='Importance sampling', capsize=10)
    plt.ylabel(label, fontsize=30)
    plt.axhline(0, ls="--", c="k")
    plt.xlabel("$\\beta$", fontsize=30)
    plt.legend()
    plt.locator_params(axis="x", nbins=7)
    plt.show()


def compare_dE_plot_ymax(
    boot_folders,
    n_repeats,
    llr_key,
    label,
    minimum=False,
    num_samples=200,
    error_type="standard deviation",
):
    colours = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
    ]
    dEsq = np.array([])
    bmax = np.array([])
    bmax_err = np.array([])
    ymax = np.array([])
    ymax_err = np.array([])
    for i, (bf, nr) in enumerate(zip(boot_folders, n_repeats)):
        full_y = np.array([])
        full_b = np.array([])
        for j in range(nr):
            full_df = pd.read_csv(f"{bf}{j}/CSV/obs_critical.csv")
            if full_df[llr_key].values.sum() == 0:
                full_df = pd.read_csv(f"{bf}{j}/CSV/obs.csv")
            y = [
                full_df.iloc[i][llr_key]
                for i in range(len(full_df))
                if not np.isnan(full_df.iloc[i][llr_key])
            ]
            b = [
                full_df.iloc[i]["b"]
                for i in range(len(full_df))
                if not np.isnan(full_df.iloc[i][llr_key])
            ]
            if minimum:
                arg = np.argmin(y)
            else:
                arg = np.argmax(y)
            full_y = np.append(full_y, y[arg])
            full_b = np.append(full_b, b[arg])
        V = full_df["V"].values[0]
        dEsq = np.append(
            dEsq,
            (2 * pd.read_csv(f"{bf}0/CSV/final.csv")["dE"].values[0] / (6 * V)) ** 2.0,
        )
        bmax_err = np.append(
            bmax_err, error.calculate_error(full_b, num_samples, error_type)
        )
        bmax = np.append(bmax, full_b.mean(axis=0))
        ymax_err = np.append(
            ymax_err, error.calculate_error(full_y, num_samples, error_type)
        )
        ymax = np.append(ymax, full_y.mean(axis=0))
        plt.errorbar(dEsq[-1], ymax[-1], ymax_err[-1], fmt=colours[i] + "o", capsize=10)
    plt.ylabel(label, fontsize=30)
    print(dEsq)
    print(ymax)
    print(ymax_err)
    ylim, ylim_err = utils.plot_extrap(dEsq, ymax, ymax_err, np.arange(len(dEsq)), "k")
    print(ylim, ylim_err)
    # plt.xlabel('$(a^4 \Delta_E/ 6\\tilde{V})^2$', fontsize = 30)
    plt.xlabel("$(\Delta_{u_p})^2$", fontsize=30)
    plt.locator_params(axis="x", nbins=7)
    plt.show()


def histogram_ymax(boot_folders, n_repeats, llr_key, label, minimum=False):
    colours = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
    ]
    dEsq = np.array([])
    bmax = np.array([])
    ymax = np.array([])
    fig, axs = plt.subplots(2, len(boot_folders), figsize=(5 * len(boot_folders), 10))
    for i, (bf, nr) in enumerate(zip(boot_folders, n_repeats)):
        full_y = np.array([])
        full_b = np.array([])
        for j in range(nr):
            full_df = pd.read_csv(f"{bf}{j}/CSV/obs_critical.csv")
            if full_df[llr_key].values.sum() == 0:
                full_df = pd.read_csv(f"{bf}{j}/CSV/obs.csv")
                print("obs.csv")
            y = [
                full_df.iloc[i][llr_key]
                for i in range(len(full_df))
                if not np.isnan(full_df.iloc[i][llr_key])
            ]
            b = [
                full_df.iloc[i]["b"]
                for i in range(len(full_df))
                if not np.isnan(full_df.iloc[i][llr_key])
            ]
            if minimum:
                arg = np.argmin(y)
            else:
                arg = np.argmax(y)
            full_y = np.append(full_y, y[arg])
            full_b = np.append(full_b, b[arg])
        V = full_df["V"].values[0]
        dEsq = np.append(
            dEsq, (pd.read_csv(f"{bf}0/CSV/final.csv")["dE"].values[0] / (6 * V)) ** 2.0
        )
        bmax = np.append(bmax, full_b.mean(axis=0))
        ymax = np.append(ymax, full_y.mean(axis=0))
        axs[0, i].hist(full_y, color=colours[i], histtype="step", density=True)
        axs[1, i].hist(full_b, color=colours[i], histtype="step", density=True)
        axs[0, i].set_xlabel(label)
        axs[1, i].set_xlabel("$\\beta$")
    plt.show()
