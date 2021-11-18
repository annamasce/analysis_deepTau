import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common.eff_rate import *
from common.dataset import Dataset
import json
import argparse
from HLT_paths import paths, Pt_thr_paths

parser = argparse.ArgumentParser()
parser.add_argument("plotName", help="name of the pdf plot")
parser.add_argument("datasetType", help="dataset type to identify proper generator selection for efficiency")
args = parser.parse_args()
if args.datasetType not in ["EleTau", "MuTau", "DiTau"]:
    sys.exit("Wrong dataset type. choose one of the following: EleTau, MuTau, TauMET, HighPtTau, DiTau")

plot_name = args.plotName
plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/"
data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/211109/"
fileName_eff = "VBFHToTauTau_deepTau.root"
fileName_rates = "Ephemeral_deepTau.root"
treeName_gen = "gen_counter"
treeName_in = "final_{}_counter".format(args.datasetType)

Pt_thr_list = [Pt_thr_paths[args.datasetType]]

# get VBF sample
print("Loading sample for efficiency")
dataset_eff = Dataset(data_path + fileName_eff, treeName_in, treeName_gen, type=args.datasetType)
taus = dataset_eff.get_taus()
taus = taus[taus.passed_last_filter > 0]
print(taus.pt)
gen_taus = dataset_eff.get_gen_taus()

# get HLT physics sample
print("Loading sample for rate")
dataset_rates = Dataset(data_path + fileName_rates, treeName_in, treeName_gen, type=args.datasetType)
taus_rates = dataset_rates.get_taus()
taus_rates = taus_rates[taus_rates.passed_last_filter > 0]
Nev_den = len(dataset_rates.get_gen_events())

fileName_eff_base = "VBFHToTauTau_oldHLT.root"
fileName_rates_base = "Ephemeral_oldHLT.root"
treeName_gen_base = "gen_counter"
treeName_in_base = "final_{}_counter".format(paths[args.datasetType])

# get VBF sample for old HLT
print("Loading sample for oldHLT efficiency")
dataset_eff_base = Dataset(data_path + fileName_eff_base, treeName_in_base, treeName_gen_base, type=args.datasetType)
taus_base = dataset_eff_base.get_taus()
taus_base = taus_base[taus_base.passed_last_filter > 0]
gen_taus_base = dataset_eff_base.get_gen_taus()

# get HLT physics sample for oldHLT
print("Loading sample for oldHLT rate")
dataset_rates_base = Dataset(data_path + fileName_rates_base, treeName_in_base, treeName_gen_base)
taus_rates_base = dataset_rates_base.get_taus()
taus_rates_base = taus_rates_base[taus_rates_base.passed_last_filter > 0]
Nev_den_base = len(dataset_rates_base.get_gen_events())


with PdfPages(plot_path + args.datasetType + '/eff_vs_rate_{}.pdf'.format(plot_name)) as pdf:

    print("Getting reasonable thresholds")
    thr_list = np.linspace(0.0, 1.0, num=100)
    
    for Pt_thr in Pt_thr_list:

        print("Pt threshold:", Pt_thr)
        print("Computing efficiencies")
        eff_list, eff_err_low, eff_err_up = compute_deepTau_eff_list_singleTau(dataset_eff, thr_list, Pt_thr=Pt_thr)
        xerr = np.zeros((2, len(thr_list)))
        xerr[0] = eff_err_low
        xerr[1] = eff_err_up
        # print(eff_list)

        print("Computing rates")
        rates, rates_err_low, rates_err_up = compute_deepTau_rate_list_singleTau(taus_rates, Nev_den, thr_list, Pt_thr=Pt_thr)
        yerr = np.zeros((2, len(thr_list)))
        yerr[0] = rates_err_low
        yerr[1] = rates_err_up

        # Compute efficiency before selection
        eff_initial = compute_base_eff_singleTau(dataset_eff, Pt_thr=Pt_thr)
        print("Efficiency before deeptau:", eff_initial)
        
        # plot eff vs rate
        plt.title("Efficiency vs Rate for Pt > {} GeV".format(Pt_thr))
        plt.xlabel("Efficiency")
        plt.ylabel("Rate [Hz]")
        # plt.xlim(0.3, 0.8)
        plt.errorbar(eff_list, rates, yerr=yerr, xerr=xerr, fmt='.--',  label="deepTau discriminator")
        plt.axvline(eff_initial[0], linestyle="--", linewidth=0.7, color="green", label="eff before deepTau producer")

        # Compute old working point
        eff_isocut, eff_isocut_err_low, eff_isocut_err_up = compute_base_eff_singleTau(dataset_eff_base, Pt_thr=Pt_thr)
        rate_isocut, rate_isocut_err_low, rate_isocut_err_up = compute_base_rate_singleTau(taus_rates_base, Nev_den_base, Pt_thr=Pt_thr)
        print("efficiency", eff_isocut, eff_isocut_err_low, eff_isocut_err_up)
        print("rate", rate_isocut, rate_isocut_err_low, rate_isocut_err_up)
        yerr_isocut = np.zeros((2, 1))
        yerr_isocut[0] = rate_isocut_err_low
        yerr_isocut[1] = rate_isocut_err_up
        xerr_isocut = np.zeros((2, 1))
        xerr_isocut[0] = eff_isocut_err_low
        xerr_isocut[1] = eff_isocut_err_up
        plt.errorbar(eff_isocut, rate_isocut, yerr=yerr_isocut, xerr=xerr_isocut, color="red", marker='.', label="Run 2 WP")

        plt.legend()
        # plt.yscale("log")
        plt.grid(True)
        pdf.savefig()
        plt.close()
