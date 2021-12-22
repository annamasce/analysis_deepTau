from common.dataset import Dataset
from common.eff_rate import *
from common.selection import DzMatchFilter
from run_optimization import get_leading_pair
import awkward as ak
from scipy.optimize import minimize, minimize_scalar
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import argparse


def true_taus_selected(taus, Pt_thr):
    tau_1 = taus[0]
    tau_2 = taus[1]
    # select only true taus that pass generator preselection
    num_tau_mask_1 = num_mask_eff(tau_1, Pt_thr=Pt_thr)
    num_tau_mask_2 = num_mask_eff(tau_2, Pt_thr=Pt_thr)
    pair_mask = num_tau_mask_1 & num_tau_mask_2
    ev_mask = ditau_selection(num_tau_mask_1, num_tau_mask_2)
    tau_1_selected = (tau_1[pair_mask])[ev_mask]
    tau_2_selected = (tau_2[pair_mask])[ev_mask]
    return tau_1_selected, tau_2_selected


def deep_thr_lin1(tau, par, Pt_thr):
    a_1 = (par[1] - par[0]) / (100 - Pt_thr)
    b_1 = par[1] - 100 * a_1
    c = 0.125
    a_2 = (c - par[1]) / 200
    b_2 = c - 300 * a_2

    thr1 = ak.where(tau.pt < 100, a_1 * tau.pt + b_1, 0)
    thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
    thr3 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2 + thr3
    return deep_thr


def deep_thr_parab(tau, par, Pt_thr):
    a_1 = (par[1] - par[0]) / (-10000 - Pt_thr ** 2 + 200 * Pt_thr)
    b_1 = -200 * a_1
    c_1 = par[1] - a_1 * 100 ** 2 - b_1 * 100

    c = 0.125
    a_2 = (c - par[1]) / 200
    b_2 = c - 300 * a_2

    thr1 = ak.where(tau.pt < 100, a_1 * tau.pt ** 2 + b_1 * tau.pt + c_1, 0)
    thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
    thr3 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2 + thr3
    return deep_thr


def deep_thr_lin2(tau, par, Pt_thr):
    c = 0.125
    m = (c - par[0]) / (300 - Pt_thr)
    q = c - m * 300
    thr1 = ak.where((tau.pt < 300), m * tau.pt + q, 0)
    thr2 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2
    return deep_thr


# define function to compute algorithmic efficiency
def compute_eff_algo(tau_1, tau_2, a, Pt_thr, deep_thr):
    eff_presel = ak.sum(ak.num(tau_1, axis=-1) >= 1)
    deepTau_mask_1 = tau_1.deepTau_VSjet > deep_thr(tau_1, a, Pt_thr)
    deepTau_mask_2 = tau_2.deepTau_VSjet > deep_thr(tau_2, a, Pt_thr)
    eff = ak.sum(ditau_selection(deepTau_mask_1, deepTau_mask_2))
    return compute_eff_witherr(eff, eff_presel)


# define function to compute rate
def compute_rate(tau_1, tau_2, Nev_den, Pt_thr, a, L1rate, deep_thr):
    mask_1 = (tau_1.deepTau_VSjet > deep_thr(tau_1, a, Pt_thr)) & reco_tau_selection(tau_1, minPt=Pt_thr)
    mask_2 = (tau_2.deepTau_VSjet > deep_thr(tau_2, a, Pt_thr)) & reco_tau_selection(tau_2, minPt=Pt_thr)
    Nev_num = ak.sum(ditau_selection(mask_1, mask_2))
    return Nev_num / Nev_den * L1rate


k = math.log(2) / 0.1


def loss(rate):
    if rate <= 46:
        return 0
    if rate > 46.1:
        return 1
    return math.exp(k * (rate - 46)) - 1


def compute_eff_algo_base(tau_1, tau_2):
    eff_presel = len(tau_1)
    mask_1 = iso_tau_selection(tau_1, "mediumIsoAbs", "mediumIsoRel")
    mask_2 = iso_tau_selection(tau_2, "mediumIsoAbs", "mediumIsoRel")
    eff = ak.sum(ditau_selection(mask_1, mask_2))
    return compute_eff_witherr(eff, eff_presel)


def compute_rate_base(tau_1, tau_2, Nev_den, Pt_thr, L1rate):
    mask_1 = iso_tau_selection(tau_1, "mediumIsoAbs", "mediumIsoRel") & reco_tau_selection(tau_1, minPt=Pt_thr)
    mask_2 = iso_tau_selection(tau_2, "mediumIsoAbs", "mediumIsoRel") & reco_tau_selection(tau_2, minPt=Pt_thr)
    Nev_num = ak.sum(ditau_selection(mask_1, mask_2))
    return Nev_num / Nev_den * L1rate


def run_optimization(taus, taus_rates, Pt_thr, deep_thr, Nev_den, L1rate):
    taus_selected = true_taus_selected(taus, Pt_thr)

    def f(a):
        rate = compute_rate(taus_rates[0], taus_rates[1], Nev_den, Pt_thr, a, L1rate, deep_thr)
        eff_algo, _, _ = compute_eff_algo(taus_selected[0], taus_selected[1], a, Pt_thr, deep_thr)
        print(a, "\trate\t:", rate, "\teff\t:", eff_algo)
        return - eff_algo + loss(rate)

    # res = minimize(f, [0.7, 0.7], bounds=((0, 1), (0, 1)), method="L-BFGS-B", options={"eps": 0.001})
    res = minimize(f, [0.571, 0.453], bounds=((0.125, 1), (0.125, 1)), method="L-BFGS-B", options={"eps": 0.001})
    # res = minimize(f, [0.7], bounds=[(0.125, 1)], method="L-BFGS-B", options={"eps": 0.001})

    print("Optimized parameters:", res.x)
    print("Rate:", compute_rate(taus_rates[0], taus_rates[1], Nev_den, Pt_thr, res.x, L1rate, deep_thr))
    return res.x


def plot_algo_eff_singleTau(taus, pt_bins, ax, deep_thr, optim_x, label):
    eff = np.zeros([3, len(pt_bins)])
    pt_arrays = np.zeros([3, len(pt_bins)])
    taus_selected = true_taus_selected(taus, Pt_thr)
    tau_leading = (taus_selected[0])[:, 0]
    tau_subleading = (taus_selected[1])[:, 0]
    for i in range(0, len(pt_bins)):

        pt_min = pt_bins[i]
        if i != len(pt_bins) - 1:
            pt_max = pt_bins[i + 1]
            central_value = (pt_max + pt_min) / 2
            pt_arrays[0, i] = central_value
            pt_arrays[1, i] = central_value - pt_min
            pt_arrays[2, i] = pt_max - central_value
            bin_mask_lead = (tau_leading.pt >= pt_min) & (tau_leading.pt < pt_max)
            bin_mask_sublead = (tau_subleading.pt >= pt_min) & (tau_subleading.pt < pt_max)
        else:
            central_value = (1000 + pt_min) / 2
            pt_arrays[0, i] = central_value
            pt_arrays[1, i] = central_value - pt_min
            pt_arrays[2, i] = 1000 - central_value
            bin_mask_lead = (tau_leading.pt >= pt_min)
            bin_mask_sublead = (tau_subleading.pt >= pt_min)

        Nev_presel = ak.sum(bin_mask_lead) + ak.sum(bin_mask_sublead)

        deepTau_mask_lead = tau_leading.deepTau_VSjet > deep_thr(tau_leading, optim_x, Pt_thr)
        deepTau_mask_sublead = tau_subleading.deepTau_VSjet > deep_thr(tau_subleading, optim_x, Pt_thr)
        Nev_num = ak.sum(bin_mask_lead & deepTau_mask_lead) + ak.sum(bin_mask_sublead & deepTau_mask_sublead)

        eff_results = compute_eff_witherr(Nev_num, Nev_presel)
        eff[:, i] = eff_results

    ax.errorbar(pt_arrays[0, :], eff[0, :],
                xerr=(pt_arrays[1, :], pt_arrays[2, :]),
                yerr=(eff[1, :], eff[2, :]), marker=".", label=label, linewidth=0.7, linestyle="")
    ax.set_xlabel(r"tau $p_{T}$ [GeV]")
    ax.set_ylabel("algo eff")
    ax.set_xscale("log")
    ax.set_xticks(pt_bins)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(30, 1000)
    ax.set_ylim(0.86, 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Tag of the pdf files with plots to be created")
    parser.add_argument("plotPath", help="Path of pdf files with plots to be created")
    args = parser.parse_args()

    tag = args.tag

    # deepTau eff dataset
    fileName = ["/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/211220/VBFHToTauTau_deepTau.root",
                "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/211220/ZprimeToTauTau_deepTau.root"]
    treeName_in = "final_DiTau_counter"
    treeName_gen = "gen_counter"
    dataset_eff = Dataset(fileName, treeName_in, treeName_gen, type="DiTau", apply_l2=True)
    taus = dataset_eff.get_taupairs(apply_selection=False)
    taus = DzMatchFilter(taus[0], taus[1])

    # deepTau rate dataset
    fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/211220/Ephemeral_deepTau.root"
    dataset_rates = Dataset(fileName_rates, treeName_in, treeName_gen, type="DiTau", apply_l2=True)
    taus_rates = dataset_rates.get_taupairs(apply_selection=False)
    taus_rates = DzMatchFilter(taus_rates[0], taus_rates[1])
    Nev_den = len(dataset_rates.get_gen_events())
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792 / 7319
    L1rate_bm = L1rate * lumi_bm / lumi_real

    pt_bins = [35, 40, 45, 50, 60, 70, 100, 150, 200, 250, 300, 400, 600]
    fig, ax = plt.subplots()

    Pt_thr = 35
    optim_x = run_optimization(taus, taus_rates, Pt_thr, deep_thr_lin1, Nev_den, L1rate_bm)
    plot_algo_eff_singleTau(taus, pt_bins, ax, deep_thr_lin1, optim_x, tag)
    plt.legend()
    # plt.savefig("algo_eff_flatten_{}.pdf".format(tag))
    plt.show()
    # plt.close()

    # func_dictionary = {
    #     "lin model 1": [deep_thr_lin1, [0.5207, 0.3357]],
    #     # "lin model 2": [deep_thr_lin2, [0.54034784]],
    #     # "parab model": [deep_thr_parab, [0.59785017, 0.39570033]]
    # }
    #
    # for func in func_dictionary:
    #     plot_algo_eff_singleTau(taus, pt_bins, ax, func_dictionary[func][0], func_dictionary[func][1], func)
    # plt.legend()
    # plt.savefig(args.plotPath + "/algo_eff_flatten_{}.pdf".format(tag))
    # plt.close()
