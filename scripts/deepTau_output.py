import sys
sys.path.append('../')
from helpers import ROC_fromTuples, getEvents_fromFile, compute_efficiency, compute_rates, cutbased_eff_flattentuples
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
fileName_eff = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/VBFToTauTau_lowPt.root"
fileName_rates = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/HLTPhys_1-8_lowPt.root"

cutbased_vars = {
                 "loose": ["tau_looseIsoAbs", "tau_looseIsoRel"],
                 "medium": ["tau_mediumIsoAbs", "tau_mediumIsoRel"],
                 "tight": ["tau_tightIsoAbs", "tau_tightIsoRel"]
                }
colors = ['green', 'red', 'orange']

with PdfPages(plot_path + 'deepTau_output_{}.pdf'.format(plot_name)) as pdf:

    # get trees from file
    treeName_gen = "gen_counter"
    treeName_in = "initial_counter"
    events_gen, events_in = getEvents_fromFile(fileName_eff, treeName_gen, treeName_in)

    # generate ROC curve for deepTau_VSjet
    fpr, tpr, thr, pred, truth = ROC_fromTuples(events_in)
    score = auc(fpr, tpr)
    print("AUC ROC:", score)
    plt.yscale('log')
    plt.xlabel(r'$\tau$ ID efficiency')
    plt.ylabel('jet misID probability')
    plt.title("deepTau ROC curve in VBF simulation")
    plt.plot(tpr, fpr, '-', label="AUC-ROC score: {}".format(round(score, 4)))
    i=0
    for key, value in cutbased_vars.items():
        tpr_cut, fpr_cut = cutbased_eff_flattentuples(events_in, value[0], value[1])
        print(key, tpr_cut, fpr_cut)
        plt.plot(tpr_cut, fpr_cut, color=colors[i], marker='.', label="{} cut id".format(key))
        i = i + 1
    plt.legend()
    pdf.savefig()
    plt.close()

    # deepTau_VSjet distribution for signal and background in VBF sample
    plt.hist(pred[truth==1].flatten(), bins=50, alpha=0.5, label=r"$\tau_h$")
    plt.hist(pred[truth==0].flatten(), bins=50, alpha=0.5, label="jets")
    plt.legend()
    plt.title("deepTau output for VBF simulation")
    plt.xlabel("deepTau_VSjet")
    pdf.savefig()
    plt.close()

    # deepTau_VSjet distribution in HLTphys sample
    events_gen_rates, events_in_rates = getEvents_fromFile(fileName_rates, treeName_gen, treeName_in)
    plt.title("deepTau output for HLT physics data")
    plt.xlabel("deepTau_VSjet")
    plt.hist(events_in_rates['deepTau_VSjet'].flatten(), bins=50, alpha=0.5)
    pdf.savefig()
    plt.close()