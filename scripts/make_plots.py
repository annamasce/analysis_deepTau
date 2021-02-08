import sys
# sys.path.append('../')
from helpers import *
import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from selection import *
from eff_rate import *

plot_name = sys.argv[1]
plot_path = '/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/minPt20/'
fileName_eff = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/VBFToTauTau_lowPt.root"
fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/HLTPhys_1-8_lowPt.root"

Pt_thr_list = [20, 25, 30, 35, 40, 45]
# Pt_thr_list = [20]

isocut_vars = {
                 "loose": ["looseIsoAbs", "looseIsoRel"],
                 "medium": ["mediumIsoAbs", "mediumIsoRel"],
                 "tight": ["tightIsoAbs", "tightIsoRel"]
                }
colors = ['green', 'red', 'orange']

with PdfPages(plot_path + 'eff_vs_rate_{}.pdf'.format(plot_name)) as pdf:

    # get trees for efficiency from file
    treeName_gen = "gen_counter"
    treeName_in = "initial_counter"
    events_gen, events_in = getEvents_fromFile(fileName_eff, treeName_gen, treeName_in)
    taus = getTaus(events_in, is_old=True)
    # L1taus = getL1taus(events_in)

    # # apply last modules removed from HLT path (take only taus with L1 match and apply event cut based on dz of jet pairs)
    # taus = L1THLTTauMatching(L1taus, taus)
    # taus = HLTJetPairDzMatchFilter(taus)

    gen_taus = getTaus(events_gen,  is_gen=True)
    
    # get trees for rates from file
    events_gen_rates, events_in_rates = getEvents_fromFile(fileName_rates, treeName_gen, treeName_in)
    taus_rates = getTaus(events_in_rates, is_old=True)
    # L1taus_rates = getL1taus(events_in_rates)

    # # apply last modules removed from HLT path (take only taus with L1 match and apply event cut based on dz of jet pairs)
    # taus_rates = L1THLTTauMatching(L1taus_rates, taus_rates)
    # taus_rates = HLTJetPairDzMatchFilter(taus_rates)


    _, _, thr_list = ROC_fromTuples(events_in, get_predictions=False) # to get reasonable thresholds
    length = len(thr_list)
    thr_list = thr_list[range(0, length, 50)]
    L1rate = 73455.34
    
    for Pt_thr in Pt_thr_list:

        print("Pt threshold:", Pt_thr)
        print("Computing efficiencies")
        eff_list = compute_eff(taus, gen_taus, thr_list, Pt_thr=Pt_thr)

        print("Computing rates")
        Nev_den = len(events_gen_rates)
        rates = compute_rates(taus_rates, Nev_den, thr_list, Pt_thr=Pt_thr)
        
        # plot eff vs rate
        plt.title("Efficiency vs Rate for Pt > {}".format(Pt_thr))
        plt.xlabel("Efficiency")
        plt.ylabel("Rate [Hz]")
        plt.plot(eff_list, rates, '.-')
        pdf.savefig()
        plt.close()

        for key, value in isocut_vars.items():
            eff_isocut = compute_isocut_eff(taus, gen_taus, value[0], value[1], Pt_thr=Pt_thr)
            rate_isocut = compute_isocut_rate(taus_rates, Nev_den, value[0], value[1], Pt_thr=Pt_thr)
            print(key, eff_isocut, rate_isocut)