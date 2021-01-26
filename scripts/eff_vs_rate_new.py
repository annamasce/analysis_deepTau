import sys
sys.path.append('../')
from helpers import ROC_fromTuples, getEvents_fromFile, compute_cutbased_eff, compute_cutbased_rates
import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from selection import *

plot_name = sys.argv[1]
plot_path = '/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/'
fileName_eff = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/VBFToTauTau_lowPt.root"
fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/HLTPhys_1-8_lowPt.root"

Pt_thr_list = [20, 25, 30, 35, 40, 45]
# Pt_thr_list = [20]

cutbased_vars = {
                 "loose": ["tau_looseIsoAbs", "tau_looseIsoRel"],
                 "medium": ["tau_mediumIsoAbs", "tau_mediumIsoRel"],
                 "tight": ["tau_tightIsoAbs", "tau_tightIsoRel"]
                }
colors = ['green', 'red', 'orange']

with PdfPages(plot_path + 'eff_vs_rate_{}.pdf'.format(plot_name)) as pdf:

    # get trees for efficiency from file
    treeName_gen = "gen_counter"
    treeName_in = "initial_counter"
    events_gen, events_in = getEvents_fromFile(fileName_eff, treeName_gen, treeName_in)
    
    # get trees for rates from file
    events_gen_rates, events_in_rates = getEvents_fromFile(fileName_rates, treeName_gen, treeName_in)

    _, _, thr_list = ROC_fromTuples(events_in, get_predictions=False) # to get reasonable thresholds
    length = len(thr_list)
    thr_list = thr_list[range(0, length, 50)]
    
    for Pt_thr in Pt_thr_list:

        print("Pt threshold:", Pt_thr)
        # compute efficiencies
        print("Computing efficiencies")
        Nev_num_list = []
        num_tau_mask = true_tau_selection(events_in) & gen_tau_selection(events_in) & reco_tau_selection(events_in, minPt=Pt_thr, eta_sel=False)
        den_tau_mask = true_tau_selection(events_gen) & gen_tau_selection(events_gen)
        Nev_den = ditau_selection(den_tau_mask).sum()

        for thr in thr_list:
            num_tau_mask_final = deepTau_selection(events_in, thr) & num_tau_mask
            Nev_num = ditau_selection(num_tau_mask_final).sum()
            Nev_num_list.append(Nev_num)
            
        eff_list = Nev_num_list/Nev_den

        # compute rates
        print("Computing rates")
        Nev_num_list = []
        Nev_den = len(events_gen_rates)
        L1rate = 73455.34
        num_tau_mask = reco_tau_selection(events_in_rates, minPt=Pt_thr)
        for thr in thr_list:
            num_tau_mask_final = deepTau_selection(events_in_rates, thr) & num_tau_mask
            Nev_num = ditau_selection(num_tau_mask_final).sum()
            Nev_num_list.append(Nev_num)
        Nev_num_list = np.array(Nev_num_list)
        rates = Nev_num_list*L1rate/Nev_den

        # plot eff vs rate
        plt.title("Efficiency vs Rate for Pt > {}".format(Pt_thr))
        plt.xlabel("Efficiency")
        plt.ylabel("Rate [Hz]")
        plt.plot(eff_list, rates, '.-')
        # print(rates[range(0, len(thr), 100)])
        pdf.savefig()
        plt.close()

        for key, value in cutbased_vars.items():
            eff_cutbased = compute_cutbased_eff(events_gen, events_in, Pt_thr, value[0], value[1])
            rate_cutbased = compute_cutbased_rates(events_in_rates, Nev_den, L1rate, Pt_thr, value[0], value[1])
            print(key, eff_cutbased, rate_cutbased)