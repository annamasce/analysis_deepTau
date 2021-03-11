import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from eff_rate import *
from dataset import dataset
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("plotName", help="name of the pdf plot")
parser.add_argument("--qcd", help="work on qcd input for rates", action="store_true")
args = parser.parse_args()

plot_name = args.plotName
plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/newPlots_CMSSW_11_2_0/"
data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/"
fileName_eff = "VBFHToTauTau.root"
fileName_rates = "EphemeralHLTPhysics_1to8.root"
QCD_fileJson = "QCD_samples.json"

Pt_thr_list = [20, 25, 30, 35, 40, 45]
# Pt_thr_list = [20]

isocut_vars = {
                 "loose": ["looseIsoAbs", "looseIsoRel"],
                 "medium": ["mediumIsoAbs", "mediumIsoRel"],
                 "tight": ["tightIsoAbs", "tightIsoRel"]
                }
colors = ["green", "red", "orange"]

# get VBF sample
treeName_gen = "gen_counter"
treeName_in = "final_counter"
dataset_eff = dataset(data_path + fileName_eff, treeName_in, treeName_gen)
taus = dataset_eff.get_taus()
gen_taus = dataset_eff.get_gen_taus()

# get taus before any selection
original_taus = dataset_eff.get_taus(apply_selection=False)

# get sample for rate computation
if args.qcd:
    # get QCD sample 
    print("Getting QCD samples")
    QCD_taus_list = []
    QCD_xs_list = []
    QCD_den_list = []
    with open(QCD_fileJson, "r") as json_file:
        samples = json.load(json_file)
        for key, value in samples.items():
            data = dataset(data_path + value[0], treeName_in, treeName_gen)
            QCD_taus_list.append(data.get_taus())
            QCD_xs_list.append(value[1])
            QCD_den_list.append(len(data.get_gen_events())) 
    print(QCD_xs_list)
else:
    # get HLT physics sample
    dataset_rates = dataset(data_path + fileName_rates, treeName_in, treeName_gen)
    taus_rates = dataset_rates.get_taus()

with PdfPages(plot_path + 'eff_vs_rate_{}.pdf'.format(plot_name)) as pdf:

    print("Getting reasonable thresholds")
    thr_list = np.linspace(0.0, 1.0, num=100)
    
    for Pt_thr in Pt_thr_list:

        print("Pt threshold:", Pt_thr)
        print("Computing efficiencies")
        eff_list, eff_err_low, eff_err_up = compute_deepTau_eff_list(taus, gen_taus, thr_list, Pt_thr=Pt_thr)
        xerr = np.zeros((2, len(thr_list)))
        xerr[0] = eff_err_low
        xerr[1] = eff_err_up
        # print(eff_list)

        print("Computing rates")
        if args.qcd:
            rates = np.zeros(len(thr_list))
            rates_err_low = np.zeros(len(thr_list))
            rates_err_up = np.zeros(len(thr_list))
            for i, QCD_taus in enumerate(QCD_taus_list):
                rates_i, err_i_low, err_i_up = compute_deepTau_rate_list(QCD_taus, QCD_den_list[i], thr_list, Pt_thr=Pt_thr, is_MC=True, xs=QCD_xs_list[i])
                # print(rates_i)
                rates = np.add(rates, rates_i)
                rates_err_low.add(rates_err_low, err_i_low)
                rates_err_up.add(rates_err_up, err_i_up)
            # print(rates)
        else:
            Nev_den = len(dataset_rates.get_gen_events())
            rates, rates_err_low, rates_err_up = compute_deepTau_rate_list(taus_rates, Nev_den, thr_list, Pt_thr=Pt_thr)
        yerr = np.zeros((2, len(thr_list)))
        yerr[0] = rates_err_low
        yerr[1] = rates_err_up

        # Compute efficiency before selection
        eff_initial = compute_base_eff(original_taus, gen_taus, Pt_thr=Pt_thr)
        print("Efficiency before deeptau:", eff_initial)
        # Compute efficiency after L1 matching and dz cut
        eff_limit = compute_base_eff(taus, gen_taus, Pt_thr=Pt_thr)
        print("Efficiency after L1 matching and dz cut:", eff_limit)
        
        # plot eff vs rate
        plt.title("Efficiency vs Rate for Pt > {} GeV".format(Pt_thr))
        plt.xlabel("Efficiency")
        plt.ylabel("Rate [Hz]")
        plt.ylim(0., 200.)
        plt.errorbar(eff_list, rates, yerr=yerr, xerr=xerr, fmt='.--',  label="deepTau discriminator")
        plt.axvline(eff_initial, linestyle="--", linewidth=0.7, color="purple", label="eff before deepTau producer")
        plt.axvline(eff_limit, linestyle="--", linewidth=0.7, color="brown", label="eff after selection")
        
        j=0
        for key, value in isocut_vars.items():
            eff_isocut, eff_isocut_err_low, eff_isocut_err_up = compute_isocut_eff(taus, gen_taus, value[0], value[1], Pt_thr=Pt_thr)
            if args.qcd:
                rate_isocut = 0
                for i, QCD_taus in enumerate(QCD_taus_list):
                    rate_i = compute_isocut_rate(QCD_taus, QCD_den_list[i], value[0], value[1], Pt_thr=Pt_thr, is_MC=True, xs=QCD_xs_list[i])
                    # print(rate_i)
                    rate_isocut = rate_isocut + rate_i
            else:
                rate_isocut, rate_isocut_err_low, rate_isocut_err_up = compute_isocut_rate(taus_rates, Nev_den, value[0], value[1], Pt_thr=Pt_thr)
            print("efficiency", key, eff_isocut, eff_isocut_err_low, eff_isocut_err_up)
            print("rate", key, rate_isocut, rate_isocut_err_low, rate_isocut_err_up)
            yerr_isocut = np.zeros((2, 1))
            yerr_isocut[0] = rate_isocut_err_low
            yerr_isocut[1] = rate_isocut_err_up
            xerr_isocut = np.zeros((2, 1))
            xerr_isocut[0] = eff_isocut_err_low
            xerr_isocut[1] = eff_isocut_err_up
            plt.errorbar(eff_isocut, rate_isocut, yerr=yerr_isocut, xerr=xerr_isocut, color=colors[j], marker='.', label="{} cut id".format(key))
            j = j + 1

        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()
