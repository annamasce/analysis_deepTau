from common.dataset import Dataset
from common.eff_rate import *
from run_optimization import get_leading_pair
import awkward as ak
from scipy.optimize import minimize, minimize_scalar
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

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


# def deep_thr(tau, par, Pt_thr):
#
#     a_1 = (par[1] - par[0]) / (100 - Pt_thr)
#     b_1 = par[1] - 100 * a_1
#     c = 0.125
#     a_2 = (c - par[1]) / 200
#     b_2 = c - 300 * a_2
#
#     thr1 = ak.where(tau.pt < 100, a_1 * tau.pt + b_1, 0)
#     thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
#     thr3 = ak.where(tau.pt >= 300, c, 0)
#     deep_thr = thr1 + thr2 + thr3
#     return deep_thr


# def deep_thr(tau, par, Pt_thr):
#     a_1 = (par[1] - par[0]) / (-10000 - Pt_thr ** 2 + 200 * Pt_thr)
#     b_1 = -200 * a_1
#     c_1 = par[1] - a_1 * 100 ** 2 - b_1 * 100
#
#     c = 0.125
#     a_2 = (c - par[1]) / 200
#     b_2 = c - 300 * a_2
#
#     thr1 = ak.where(tau.pt < 100, a_1 * tau.pt ** 2 + b_1 * tau.pt + c_1, 0)
#     thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
#     thr3 = ak.where(tau.pt >= 300, c, 0)
#     deep_thr = thr1 + thr2 + thr3
#     return deep_thr

def deep_thr(tau, par, Pt_thr):
    c = 0.125
    m = (c - par[0])/(300 - Pt_thr)
    q = c - m*300
    thr1 = ak.where((tau.pt < 300), m*tau.pt + q, 0)
    thr2 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2
    return deep_thr


# define function to compute algorithmic efficiency
def compute_eff_algo(tau_1, tau_2, a, Pt_thr):
    eff_presel = ak.sum(ak.num(tau_1, axis=-1))
    deepTau_thr_1 = deep_thr(tau_1, a, Pt_thr)
    deepTau_thr_2 = deep_thr(tau_2, a, Pt_thr)
    deepTau_mask_1 = tau_1.deepTau_VSjet > deepTau_thr_1
    deepTau_mask_2 = tau_2.deepTau_VSjet > deepTau_thr_2
    eff = ak.sum(ditau_selection(deepTau_mask_1, deepTau_mask_2))
    return compute_eff_witherr(eff, eff_presel)


# define function to compute rate
def compute_rate(tau_1, tau_2, Nev_den, Pt_thr, a, L1rate):
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

if __name__ == '__main__':

    tag = "lin2"
    fileName = [
                "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/VBFHToTauTau.root",
                "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/ZprimeToTauTau.root"
               ]
    treeName_in = "final_counter"
    treeName_gen = "gen_counter"
    # Pt_thr = 30
    dataset_eff = Dataset(fileName, treeName_in, treeName_gen)
    taus = dataset_eff.get_taus()

    fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/EphemeralHLTPhysics_1to8.root"
    dataset_rates = Dataset(fileName_rates, treeName_in, treeName_gen)
    taus_rates = dataset_rates.get_taus()
    Nev_den = len(dataset_rates.get_gen_events())
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792 / 7319
    L1rate_bm = L1rate * lumi_bm / lumi_real

    Pt_thr_list = [30, 35]
    pt_bins = [30, 35, 40, 45, 50, 60, 70, 100, 150, 200, 250, 300, 400, 500]
    eff = np.zeros([len(Pt_thr_list), 3, len(pt_bins)])
    eff_base = np.zeros([3, len(pt_bins)])
    optim_res = []
    colors = ["green", "blue", "violet"]
    fig, ax = plt.subplots()

    for index, Pt_thr in enumerate(Pt_thr_list):

        taus_selected = true_taus_selected(taus, Pt_thr)
        def f(a):
            rate = compute_rate(taus_rates[0], taus_rates[1], Nev_den, Pt_thr, a, L1rate_bm)
            # print("rate\t:", rate)
            eff_algo, _, _ = compute_eff_algo(taus_selected[0], taus_selected[1], a, Pt_thr)
            # print("eff\t:", eff_algo)
            # print("funct\t:", -eff_algo + loss(rate), "\n")
            return - eff_algo + loss(rate)

        # res = minimize(f, [0.7, 0.7], bounds=((0, 1), (0, 1)), method="L-BFGS-B", options={"eps": 0.01})
        # res = minimize(f, [0.7, 0.5], bounds=((0.125, 1), (0.125, 1)), method="L-BFGS-B", options={"eps": 0.001})
        res = minimize(f, [0.7], bounds=[(0.125, 1)], method="L-BFGS-B", options={"eps": 0.001})

        print("Optimized parameters:", res.x)
        optim_res.append(res.x)
        # print(0.4-100*res.x)


        # pt_bins = [35, 40, 45, 50, 55, 60, 70, 100, 169, 221, 279, 343, 414, 491]

        ### TO FIX!
        if Pt_thr == 35:
            start = 1
        else:
            start = 0

        for i in range(start, len(pt_bins)):
            # pt_min = pt_bins[i]
            # if i != len(pt_bins) - 1:
            #     pt_max = pt_bins[i + 1]
            #     bin_mask_1 = (taus_selected[0].pt>=pt_min) & (taus_selected[0].pt<pt_max)
            #     bin_mask_2 = (taus_selected[1].pt>=pt_min) & (taus_selected[1].pt<pt_max)
            # else:
            #     bin_mask_1 = (taus_selected[0].pt>=pt_min)
            #     bin_mask_2 = (taus_selected[1].pt>=pt_min)
            # pair_mask = bin_mask_1 & bin_mask_2
            # ev_mask = ditau_selection(bin_mask_1, bin_mask_2)
            # tau_1 = ((taus_selected[0])[pair_mask])[ev_mask]
            # tau_2 = ((taus_selected[1])[pair_mask])[ev_mask]
            # eff_results = compute_eff_algo(tau_1, tau_2, res.x, Pt_thr)
            # eff[index, :, i] = eff_results
            # eff_base_results = compute_eff_algo_base(tau_1, tau_2)
            # eff_base[:, i] = eff_base_results

            tau_leading = (taus_selected[0])[:, 0]
            # print(tau_leading.pt[range(5)])
            tau_subleading = (taus_selected[1])[:, 0]
            # print(tau_subleading.pt[range(5)])
            pt_min = pt_bins[i]
            if i != len(pt_bins) - 1:
                pt_max = pt_bins[i + 1]
                bin_mask_lead = (tau_leading.pt >= pt_min) & (tau_leading.pt < pt_max)
                bin_mask_sublead = (tau_subleading.pt >= pt_min) & (tau_subleading.pt < pt_max)
            else:
                bin_mask_lead = (tau_leading.pt >= pt_min)
                bin_mask_sublead = (tau_subleading.pt >= pt_min)
            N_lead_inbin = ak.sum(bin_mask_lead)
            N_sublead_inbin = ak.sum(bin_mask_sublead)
            Nev_presel = N_lead_inbin + N_sublead_inbin

            deepTau_mask_lead = tau_leading.deepTau_VSjet > deep_thr(tau_leading, res.x, Pt_thr)
            deepTau_mask_sublead = tau_subleading.deepTau_VSjet > deep_thr(tau_subleading, res.x, Pt_thr)
            N_lead_deep = ak.sum(bin_mask_lead & deepTau_mask_lead)
            N_sublead_deep = ak.sum(bin_mask_sublead & deepTau_mask_sublead)
            Nev_num = N_lead_deep + N_sublead_deep

            base_mask_lead = iso_tau_selection(tau_leading, "mediumIsoAbs", "mediumIsoRel")
            base_mask_sublead = iso_tau_selection(tau_subleading, "mediumIsoAbs", "mediumIsoRel")
            N_lead_base = ak.sum(bin_mask_lead & base_mask_lead)
            N_sublead_base = ak.sum(bin_mask_sublead & base_mask_sublead)
            Nev_num_base = N_lead_base + N_sublead_base

            eff_results = compute_eff_witherr(Nev_num, Nev_presel)
            eff[index, :, i] = eff_results
            eff_base_results = compute_eff_witherr(Nev_num_base, Nev_presel)
            eff_base[:, i] = eff_base_results

        ax.errorbar(pt_bins[start:], eff[index, 0, start:], yerr=(eff[index, 1, start:], eff[index, 2, start:]), color=colors[index], marker="o", label="deepTau at {} GeV".format(Pt_thr), linewidth=0.7)
    ax.errorbar(pt_bins[1:], eff_base[0, 1:], yerr=(eff_base[1, start:], eff_base[2, 1:]), color="red", marker="o", label="cut-based medium", linewidth=0.7)
    ax.set_xlabel(r"lower edge $p_{T}$ bin [GeV]")
    ax.set_ylabel("algo eff")
    ax.set_xscale("log")
    ax.set_xticks(pt_bins)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    # plt.ylim(0.66, 1.1)
    plt.legend()
    plt.savefig("algo_eff_flatten_{}.pdf".format(tag))
    plt.close()

    for index, Pt_thr in enumerate(Pt_thr_list):
        tau = taus_selected[0]
        plt.plot(ak.flatten(tau.pt), ak.flatten(deep_thr(tau, optim_res[index], Pt_thr)), '.')
        plt.xlim(left=0, right=500)
        plt.xlabel(r"leading tau $p_{T}$ [GeV]")
        plt.ylabel("deepTau thr")
        plt.savefig("deep_thr_function_{}_{}.pdf".format(Pt_thr, tag))
        plt.close()