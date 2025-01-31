import sys
import argparse
import json
from functools import partial
from common.dataset import Dataset
from common.eff_rate import *
import awkward as ak
from scipy.optimize import minimize, minimize_scalar
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from HLT_paths import rate_bm_paths, Pt_thr_paths


def true_taus_selected(datasets, Pt_thr):
    # select only true taus that pass generator preselection
    taus_list = []
    for dataset in datasets:
        _, good_events = dataset.evt_base_selection()
        print(good_events)
        taus = dataset.get_taus()
        num_tau_mask = (taus.passed_last_filter > 0) & num_mask_eff(taus, Pt_thr=Pt_thr)
        if dataset.type in ["TauMET", "HighPtTau"]:
            ev_mask = (ak.sum(num_tau_mask, axis=-1) == 1) & good_evt_selection(dataset.get_events(), good_events)
        else:
            ev_mask = (ak.sum(num_tau_mask, axis=-1) > 0) & good_evt_selection(dataset.get_events(), good_events)
        taus_selected = (taus[num_tau_mask])[ev_mask]
        taus_list.append(taus_selected)
    taus_array_final = ak.concatenate(taus_list)
    print(len(taus_array_final))
    return taus_array_final


# define function to compute algorithmic efficiency
def compute_eff_algo(taus, a, Pt_thr, deep_thr, optimise_VSe=False):
    eff_presel = ak.sum(ak.num(taus, axis=-1) > 0)
    if optimise_VSe:
        deepTau_mask = (taus.deepTau_VSjet > deep_thr(taus, a[1:], Pt_thr)) & (taus.deepTau_VSe > deep_thr_VSele(taus, a[0]))
    else:
        deepTau_mask = taus.deepTau_VSjet > deep_thr(taus, a, Pt_thr)
    eff = ak.sum(ak.sum(deepTau_mask, axis=-1) > 0)
    return compute_eff_witherr(eff, eff_presel)


# define function to compute rate
def compute_rate(taus, Nev_den, Pt_thr, a, L1rate, deep_thr, optimise_VSe=False):
    if optimise_VSe:
        mask = (taus.deepTau_VSjet > deep_thr(taus, a[1:], Pt_thr)) & (taus.deepTau_VSe > deep_thr_VSele(taus, a[0])) & reco_tau_selection(taus, minPt=Pt_thr)
    else:
        mask = (taus.deepTau_VSjet > deep_thr(taus, a, Pt_thr)) & reco_tau_selection(taus, minPt=Pt_thr)
    Nev_num = ak.sum(ak.sum(mask, axis=-1) > 0)
    return Nev_num / Nev_den * L1rate


k = math.log(2) / 0.1


def loss(rate, rate_bm):
    if rate <= rate_bm:
        return 0
    if rate > (rate_bm + 0.1):
        return 1
    return math.exp(k * (rate - rate_bm)) - 1


def run_optimization(taus_selected, taus_rates, rate_bm, Pt_thr, deep_thr, Nev_den, L1rate, optimise_VSe=False):
    def f(a):
        rate = compute_rate(taus_rates, Nev_den, Pt_thr, a, L1rate, deep_thr, optimise_VSe=optimise_VSe)
        eff_algo, _, _ = compute_eff_algo(taus_selected, a, Pt_thr, deep_thr, optimise_VSe=optimise_VSe)
        print(a, "\trate\t:", rate, "\teff\t:", eff_algo)
        return - eff_algo + loss(rate, rate_bm)

    # res = minimize(f, [0.4, 0.7, 0.5], bounds=((0,1), (0, 1), (0, 1)), method="L-BFGS-B", options={"eps": 0.01})
    res = minimize(f, [0.9, 0.5], bounds=((0, 1), (0, 1)), method="L-BFGS-B", options={"eps": 0.01})
    # res = minimize(f, [0.77], bounds=[(0.125, 1)], method="L-BFGS-B", options={"eps": 0.001})

    print("Optimized parameters:", res.x)
    print("Rate:", compute_rate(taus_rates, Nev_den, Pt_thr, res.x, L1rate, deep_thr, optimise_VSe=optimise_VSe))
    return res.x


def plot_algo_eff_singleTau(taus_selected, pt_bins, ax, deep_thr, optim_x, label, optimise_VSe=False, json_file_path="algo_eff.json"):
    eff = np.zeros([3, len(pt_bins)])
    pt_arrays = np.zeros([3, len(pt_bins)])
    for i in range(0, len(pt_bins)):

        pt_min = pt_bins[i]
        if i != len(pt_bins) - 1:
            pt_max = pt_bins[i + 1]
            central_value = (pt_max + pt_min) / 2
            pt_arrays[0, i] = central_value
            pt_arrays[1, i] = central_value - pt_min
            pt_arrays[2, i] = pt_max - central_value
            bin_mask = (taus_selected.pt >= pt_min) & (taus_selected.pt < pt_max)
        else:
            central_value = (3000 + pt_min) / 2
            pt_arrays[0, i] = central_value
            pt_arrays[1, i] = central_value - pt_min
            pt_arrays[2, i] = 3000 - central_value
            bin_mask = (taus_selected.pt >= pt_min)

        Nev_presel = ak.sum(ak.flatten(bin_mask, axis=-1))

        if optimise_VSe:
            deepTau_mask = (taus_selected.deepTau_VSjet > deep_thr(taus_selected, optim_x[1:], Pt_thr)) & (taus_selected.deepTau_VSe > deep_thr_VSele(taus_selected, optim_x[0]))
        else:
            deepTau_mask = taus_selected.deepTau_VSjet > deep_thr(taus_selected, optim_x, Pt_thr)
        Nev_num = ak.sum(ak.flatten(bin_mask & deepTau_mask, axis=-1))

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
    eff_dict = {
        "pt_bins": list(pt_bins),
        "eff": list(eff[0, :]),
        "err_low": list(eff[1, :]),
        "err_up": list(eff[2, :])
    }
    print(json_file_path)
    with open(json_file_path, "w") as file:
        json.dump(eff_dict, file, indent=4)

    # ax.set_xlim(30, 1000)
    # ax.set_ylim(0.86, 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="Tag of the pdf files with plots to be created")
    parser.add_argument("plotPath", help="Path of pdf files with plots to be created")
    parser.add_argument("datasetType", help="dataset type to identify proper generator selection for efficiency")
    parser.add_argument("--remove_l2", help="remove L2 filter", action="store_true")
    parser.add_argument("--optimise_VSe", help="apply L2 filter", action="store_true")
    args = parser.parse_args()

    if args.datasetType not in ["EleTau", "MuTau", "DiTau", "TauMET", "HighPtTau"]:
        sys.exit("Wrong dataset type. choose one of the following: EleTau, MuTau, TauMET, HighPtTau, DiTau")

    tag = args.tag
    plot_path = args.plotPath
    Pt_thr = Pt_thr_paths[args.datasetType]
    rate_bm = rate_bm_paths[args.datasetType]
    apply_l2 = not args.remove_l2

    # deepTau eff dataset
    fileName_1 = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/220330/VBFHToTauTau_deepTau.root"
    fileName_2 = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/220330/ZprimeToTauTau_deepTau.root"
    # fileName_3 = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/221213/WJetsToLNu_deepTau.root"
    treeName_in = "final_{}_counter".format(args.datasetType)
    treeName_gen = "gen_counter"
    datasets_eff = [Dataset(fileName_1, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2), Dataset(fileName_2, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2)]
    # datasets_eff = [Dataset(fileName_1, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2),
    #                 Dataset(fileName_2, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2),
    #                 Dataset(fileName_3, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2)]
    taus_selected = true_taus_selected(datasets_eff, Pt_thr)

    # deepTau rate dataset
    fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/220409/Ephemeral_deepTau.root"
    dataset_rates = Dataset(fileName_rates, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2)
    taus_rates = dataset_rates.get_taus()
    taus_rates = taus_rates[taus_rates.passed_last_filter]
    Nev_den = len(dataset_rates.get_gen_events())
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792 / 7319
    L1rate_bm = L1rate * lumi_bm / lumi_real

    fig, ax = plt.subplots()
    # pt_bins = [Pt_thr, 35, 40, 45, 50, 60, 70, 100, 150, 200, 250, 300, 400, 600]
    # pt_bins = [Pt_thr, 60, 70, 100, 150, 200, 250, 300, 400, 600]
    pt_bins = [Pt_thr, 200, 250, 300, 400, 500, 700, 1000, 2000]

    optim_x = run_optimization(taus_selected, taus_rates, rate_bm, Pt_thr, deep_thr_lin1_lowThr, Nev_den, L1rate_bm, optimise_VSe=args.optimise_VSe)
    plot_algo_eff_singleTau(taus_selected, pt_bins, ax, partial(deep_thr_lin2_highPt, Pt_step=500), optim_x, tag, optimise_VSe=args.optimise_VSe, json_file_path=plot_path+"/algo_eff_flatten_{}.json".format(tag))
    plt.legend()
    plt.show()
    plt.close()

    # func_dictionary = {
    #     "lin model 1, Pt steps at 300 and 500 GeV": [deep_thr_lin1_tauMET, [0.9430, 0.9299]]
    # }
    #
    # for func in func_dictionary:
    #     plot_algo_eff_singleTau(taus_selected, pt_bins, ax, func_dictionary[func][0], func_dictionary[func][1], func)
    # plt.legend()
    # plt.savefig(plot_path + "/algo_eff_flatten_{}.pdf".format(tag))
    # plt.close()

