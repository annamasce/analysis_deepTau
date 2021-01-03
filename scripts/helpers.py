import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak

def where(condition, if_true, if_false):
    return condition*if_true + (1 - condition)*if_false

def getEvents_fromFile(fileName, treeName_gen, treeName_in):
    tree_gen = uproot.open(fileName)[treeName_gen] 
    tree_in = uproot.open(fileName)[treeName_in] 
    events_gen = ak.Table(tree_gen.arrays(namedecode="utf-8")) # generator events
    events_in = ak.Table(tree_in.arrays(namedecode="utf-8")) # filtered events

    return events_gen, events_in

def ROC_fromTuples(events_in, get_predictions=True):

    gen_match = events_in["lepton_gen_match"] # generator info
    pred_all = events_in["deepTau_VSjet"]
    truth_all = where(gen_match==5, 1, 0)
    sel_tauorjets = where(gen_match>=5, 1, 0)>0
    pred = pred_all[sel_tauorjets] # deepTau predictions for tau_h and jets
    truth = truth_all[sel_tauorjets] # truth info for tau_h and jets

    fpr, tpr, thr = roc_curve(truth.flatten(), pred.flatten())

    if get_predictions:
        final_tuple = (fpr, tpr, thr, pred, truth)
    else:
        final_tuple = (fpr, tpr, thr)

    return final_tuple

def compute_efficiency(events_in, Nev_tot, thr, Pt_thr):
    mask = (events_in["lepton_gen_match"]==5) & (events_in['gen_tau_pt']>35.) & (events_in['tau_pt']>Pt_thr)
    true_taus_pred = events_in['deepTau_VSjet'][mask] # deepTau prediction for tau_h with gen Pt>20 and reco Pt>Pt_thr
    Nev_passed = np.array([((true_taus_pred>=x).sum()>=2).sum() for x in thr]).astype(np.float)
    # print(Nev_passed)
    eff = Nev_passed/Nev_tot
    return eff

def compute_rates(events_in, Nev_tot, L1rate, thr, Pt_thr):
    mask = events_in['tau_pt']>Pt_thr
    pred = events_in["deepTau_VSjet"][mask]
    passed_values = np.array([((pred>=x).sum()>=2).sum() for x in thr]).astype(np.float)
    rates = passed_values*L1rate/Nev_tot
    return rates 