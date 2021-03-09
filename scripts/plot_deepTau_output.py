import sys
sys.path.append('../')
from helpers import *
from selection import *
import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataset import dataset

plot_name = sys.argv[1]
plot_path = '/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/newPlots_CMSSW_11_2_0/'
fileName_eff = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/VBFHToTauTau.root"
treeName_gen = "gen_counter"
treeName_in = "final_counter"

cutbased_vars = {
                 "loose": ["looseIsoAbs", "looseIsoRel"],
                 "medium": ["mediumIsoAbs", "mediumIsoRel"],
                 "tight": ["tightIsoAbs", "tightIsoRel"]
                }
colors = ['green', 'red', 'orange']

with PdfPages(plot_path + 'deepTau_output_{}.pdf'.format(plot_name)) as pdf:

    # get trees from file
    dataset_VBF = dataset(fileName_eff, treeName_in, treeName_gen)
    L2taus = dataset_VBF.get_taus(apply_selection=True)

    # generate ROC curve for deepTau_VSjet
    fpr, tpr, thr, pred, truth = ROC_fromTuples(L2taus)
    score = auc(fpr, tpr)
    print("AUC ROC:", score)
    plt.yscale('log')
    plt.xlabel(r'$\tau$ ID efficiency')
    plt.ylabel('jet misID probability')
    plt.title("deepTau ROC curve in VBF simulation")
    plt.plot(tpr, fpr, '-', label="AUC-ROC score: {}".format(round(score, 4)))
    i=0
    for key, value in cutbased_vars.items():
        tpr_cut, fpr_cut = cutbased_eff_flattentuples(L2taus, value[0], value[1])
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
    plt.ylabel("counts")
    pdf.savefig()
    plt.close()