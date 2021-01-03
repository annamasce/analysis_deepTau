import sys
sys.path.append('../')
from helpers import ROC_fromTuples, getEvents_fromFile, compute_efficiency, compute_rates, where
import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plot_name = sys.argv[1]
plot_path = '/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/'

Pt_thr_list = [35, 40, 45]

with PdfPages(plot_path + 'eff_vs_rate_{}.pdf'.format(plot_name)) as pdf:

     # get trees for efficiency from file
    fileName_eff = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/VBFToTauTau_lowPt.root"
    treeName_gen = "gen_counter"
    treeName_in = "initial_counter"
    events_gen, events_in = getEvents_fromFile(fileName_eff, treeName_gen, treeName_in)
    
    # get trees for rates from file
    fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/HLTPhys_1-8.root"
    events_gen_rates, events_in_rates = getEvents_fromFile(fileName_rates, treeName_gen, treeName_in)

    fpr, tpr, thr = ROC_fromTuples(events_in, get_predictions=False) # to get reasonable thrasholds
    
    for Pt_thr in Pt_thr_list:

        # compute efficiencies
        eff = compute_efficiency(events_in, len(events_gen), thr, Pt_thr)
        
        # compute rates
        Ntot = len(events_gen_rates)
        L1rate = 73455.34
        rates = compute_rates(events_in_rates, Ntot, L1rate, thr, Pt_thr)

        # plot eff vs rate
        plt.title("Efficiency vs Rate for Pt > {}".format(Pt_thr))
        plt.xlabel("Efficiency")
        plt.ylabel("Rate [Hz]")
        plt.title("Rate vs Efficiency")
        plt.plot(eff[range(0, len(thr), 100)], rates[range(0, len(thr), 100)], '.-')
        # print(rates[range(0, len(thr), 100)])
        pdf.savefig()
        plt.close()