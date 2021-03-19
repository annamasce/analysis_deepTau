import sys

sys.path.append('../')
import argparse

import awkward as ak
import matplotlib.pyplot as plt
from scipy import optimize

from common.dataset import Dataset
from common.eff_rate import *


def deepTau_selection_ptdep(tau_array, pt_bins):
    # initialize the tau mask to False -> no tau selected
    mask = ak.broadcast_arrays(tau_array.pt, False)[1]
    for i in range(0, len(pt_bins)):
        pt_min = pt_bins[i]
        if i != len(pt_bins)-1:
            pt_max = pt_bins[i + 1]
            bin_mask = (tau_array.pt >= pt_min) & (tau_array.pt < pt_max)
            deep_tau_mask = tau_array.deepTau_VSjet > deep_thr[i]
            final_mask = bin_mask & deep_tau_mask
        else:
            final_mask = tau_array.pt >= pt_min
        mask = mask | final_mask
    return mask

def get_leading_pair(taus):
    tau_1 = taus[0]
    tau_2 = taus[1]
    # select only true taus that pass generator preselection
    num_tau_mask_1 = num_mask_eff(tau_1, Pt_thr=20)
    num_tau_mask_2 = num_mask_eff(tau_2, Pt_thr=20)
    pair_mask = num_tau_mask_1 & num_tau_mask_2
    ev_mask = ditau_selection(num_tau_mask_1, num_tau_mask_2)
    tau_1_selected = (tau_1[pair_mask])[ev_mask]
    tau_2_selected = (tau_2[pair_mask])[ev_mask]
    # get arrays of leading and subleading taus
    tau_leading = ak.firsts(tau_1_selected, axis=-1)
    tau_subleading = ak.firsts(tau_2_selected, axis=-1)
    return tau_leading, tau_subleading


plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/newPlots_CMSSW_11_2_0/"
fileName_eff = [
    "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/VBFHToTauTau.root",
    "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/ZprimeToTauTau.root"]
fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/EphemeralHLTPhysics_1to8.root"
treeName_in = "final_counter"
treeName_gen = "gen_counter"

L1rate = 75817.94
lumi_bm = 2e-2
lumi_real = 122.792 / 7319
L1rate_bm = L1rate * lumi_bm / lumi_real

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("plotName", help="tag added to the name of saved pdf plot")
    args = parser.parse_args()

    plot_name = args.plotName

    # load taus from VBF dataset
    dataset_eff = Dataset(fileName_eff, treeName_in, treeName_gen)
    taus = dataset_eff.get_taus()
    tau_leading, tau_subleading = get_leading_pair(taus)

    pt_bins = [20, 35, 45, 55, 70, 100, 300]
    ratio = []
    deep_thr = []
    for i in range(0, len(pt_bins) - 1):
        pt_min = pt_bins[i]
        pt_max = pt_bins[i + 1]
        bin_mask = (tau_leading.pt > pt_min) & (tau_leading.pt < pt_max) & (tau_subleading.pt > pt_min) & (
                    tau_subleading.pt < pt_max)
        eff_presel = ak.sum(bin_mask)

        def f(deepTau_thr):
            deepTau_mask = (tau_leading.deepTau_VSjet > deepTau_thr) & (tau_subleading.deepTau_VSjet > deepTau_thr)
            eff = ak.sum(bin_mask & deepTau_mask)
            return eff / eff_presel - 0.95

        solution = optimize.root_scalar(f, bracket=[0, 1], method='bisect')
        print(solution)
        thr = solution.root
        deepTau_mask = (tau_leading.deepTau_VSjet > thr) & (tau_subleading.deepTau_VSjet > thr)
        eff = ak.sum(bin_mask & deepTau_mask)
        ratio.append(eff / eff_presel)
        deep_thr.append(thr)
    print(ratio)
    print(deep_thr)

    # load taus from HLT physics dataset
    dataset_rates = Dataset(fileName_rates, treeName_in, treeName_gen)
    taus_rates = dataset_rates.get_taus()
    tau_1_rates = taus_rates[0]
    tau_2_rates = taus_rates[1]

    Nev_den = len(dataset_rates.get_gen_events())
    Pt_thr_list = np.linspace(30, 101, 35)
    rate_list = []
    yerr = np.zeros((2, len(Pt_thr_list)))
    for i, Pt_thr in enumerate(Pt_thr_list):
        tau_1_mask = deepTau_selection_ptdep(tau_1_rates, pt_bins) & reco_tau_selection(tau_1_rates, minPt=Pt_thr)
        tau_2_mask = deepTau_selection_ptdep(tau_2_rates, pt_bins) & reco_tau_selection(tau_2_rates, minPt=Pt_thr)
        Nev_num = ak.sum(ditau_selection(tau_1_mask, tau_2_mask))
        rate, rate_err_low, rate_err_up = compute_rate_witherr(Nev_num, Nev_den, is_MC=False, L1rate=L1rate_bm)
        rate_list.append(rate)
        yerr[0][i] = rate_err_low
        yerr[1][i] = rate_err_up
    print(rate_list)
    plt.errorbar(Pt_thr_list, rate_list, yerr=yerr, fmt='.--')
    plt.ylabel("Rate [Hz]")
    plt.xlabel(r"$p_{T}$ threshold [GeV]")
    plt.savefig(plot_path + "rateVSpt_" + plot_name + ".pdf")
