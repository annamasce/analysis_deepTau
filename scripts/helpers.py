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

def cutbased_eff_flattentuples(events_in, var_abs, var_rel):
    # cut-based selection
    cut = (events_in[var_abs]>0) | (events_in[var_rel]>0)
    # true positive selection: cut-based passed + true taus selection
    tau_sel = cut & (events_in["lepton_gen_match"]==5)
    # false positive selection: cut-based passed + jets selection
    jet_sel = cut & (events_in["lepton_gen_match"]==6)
    tpr = (tau_sel.flatten().sum())/((events_in["lepton_gen_match"]==5).flatten().sum())
    fpr = (jet_sel.flatten().sum())/((events_in["lepton_gen_match"]==6).flatten().sum())
    return tpr, fpr


def compute_efficiency(events_gen, events_in, thr, Pt_thr):
    gen_minPt = 20
    gen_maxEta = 2.1
    tau_mask_den = (events_gen["lepton_gen_match"]==5) & (events_gen["gen_tau_pt"]>gen_minPt) & (events_gen["gen_tau_eta"]<gen_maxEta) & (events_gen["gen_tau_eta"]> -gen_maxEta)
    tau_mask_num = (events_in["lepton_gen_match"]==5) & (events_in["gen_tau_pt"]>gen_minPt) & (events_in["gen_tau_eta"]<gen_maxEta) & (events_in["gen_tau_eta"]> -gen_maxEta) & (events_in["tau_pt"]>Pt_thr)
    true_taus_pred = events_in["deepTau_VSjet"][tau_mask_num] # deepTau prediction for tau_h with gen Pt>20, gen eta > 20 and reco Pt>Pt_thr
    Nev_passed = np.array([((true_taus_pred>=x).sum()>=2).sum() for x in thr]).astype(np.float)
    #  Nev_initial = (events_gen["gen_tau_pt"][tau_mask_den].counts>=2).sum()
    Nev_initial = (tau_mask_den.sum()>=2).sum()
    # print(Nev_passed)
    eff = Nev_passed/Nev_initial
    return eff

def compute_cutbased_eff(events_gen, events_in, Pt_thr, var_abs, var_rel):
    gen_minPt = 20
    gen_maxEta = 2.1
    tau_mask_den = (events_gen["lepton_gen_match"]==5) & (events_gen["gen_tau_pt"]>gen_minPt) & (events_gen["gen_tau_eta"]<gen_maxEta) & (events_gen["gen_tau_eta"]> -gen_maxEta)
    tau_mask_num = (events_in["lepton_gen_match"]==5) & (events_in["gen_tau_pt"]>gen_minPt) & (events_in["gen_tau_eta"]<gen_maxEta) & (events_in["gen_tau_eta"]> -gen_maxEta) & (events_in["tau_pt"]>Pt_thr)
    iso_cut = (events_in[var_abs]>0) | (events_in[var_rel]>0)
    mask_num = tau_mask_num & iso_cut
    Nev_passed = (mask_num.sum()>=2).sum()
    Nev_initial = (tau_mask_den.sum()>=2).sum()
    #  Nev_initial = (events_gen["gen_tau_pt"][tau_mask_den].counts>=2).sum()
    # print(Nev_passed)
    eff = Nev_passed/Nev_initial
    return eff

def compute_rates(events_in, Nev_tot, L1rate, thr, Pt_thr):
    reco_maxEta = 2.1
    mask = (events_in["tau_pt"]>Pt_thr) & (events_in["tau_eta"]<reco_maxEta) & (events_in["tau_eta"]> -reco_maxEta)
    pred = events_in["deepTau_VSjet"][mask]
    passed_values = np.array([((pred>=x).sum()>=2).sum() for x in thr]).astype(np.float)
    rates = passed_values*L1rate/Nev_tot
    return rates 

def compute_cutbased_rates(events_in, Nev_tot, L1rate, Pt_thr, var_abs, var_rel):
    reco_maxEta = 2.1
    reco_sel = (events_in["tau_pt"]>Pt_thr) & (events_in["tau_eta"]<reco_maxEta) & (events_in["tau_eta"]> -reco_maxEta)
    iso_cut = (events_in[var_abs]>0) | (events_in[var_rel]>0)
    num_sel = reco_sel & iso_cut
    Nev_passed = (num_sel.sum()>=2).sum()
    rate = Nev_passed*L1rate/Nev_tot
    return rate