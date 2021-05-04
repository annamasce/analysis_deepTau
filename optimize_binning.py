import sys

sys.path.append('../')
import argparse

import awkward as ak
import matplotlib.pyplot as plt
from scipy import optimize

from common.dataset import Dataset
from common.eff_rate import *
from run_optimization import get_leading_pair


plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/newPlots_CMSSW_11_2_0/"
fileName_eff = [
    "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/VBFHToTauTau.root",
    "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/ZprimeToTauTau.root"]
treeName_in = "final_counter"
treeName_gen = "gen_counter"

L1rate = 75817.94
lumi_bm = 2e-2
lumi_real = 122.792 / 7319
L1rate_bm = L1rate * lumi_bm / lumi_real

def eff_presel_inbin(pt_min, pt_max):
    bin_mask = (tau_leading.pt > pt_min) & (tau_leading.pt < pt_max) & (tau_subleading.pt > pt_min) & (tau_subleading.pt < pt_max)
    eff_presel = ak.sum(bin_mask)
    return eff_presel

if __name__ == '__main__':

    # load taus from VBF and Zprime dataset
    dataset_eff = Dataset(fileName_eff, treeName_in, treeName_gen)
    taus = dataset_eff.get_tau_pairs()
    tau_leading, tau_subleading = get_leading_pair(taus)

    n_bins = 100
    eff_mean = 400
    pt_min = 20
    pt_bins = [20]

    for i in range(n_bins-1):
        if eff_presel_inbin(pt_min, 2000) <= eff_mean:
            break
        def f(pt_max):
            eff_presel = eff_presel_inbin(pt_min, pt_max)
            return eff_presel - eff_mean

        solution = optimize.root_scalar(f, bracket=[pt_min, 2000], method='bisect')
        pt_max = solution.root
        print(eff_presel_inbin(pt_min, pt_max))
        pt_bins.append(pt_max)
        pt_min = pt_max

    print(eff_presel_inbin(pt_min, 2000))
    print([int(pt) for pt in pt_bins])