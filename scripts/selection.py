import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak

def true_tau_selection(events_in):
    mask = events_in["lepton_gen_match"]==5
    return mask

def gen_tau_selection(events_in, gen_minPt=20, gen_maxEta=2.1):
    mask = (events_in["gen_tau_pt"]>gen_minPt) & (events_in["gen_tau_eta"]<gen_maxEta) & (events_in["gen_tau_eta"]> -gen_maxEta)
    return mask

def reco_tau_selection(events_in, minPt=20, eta_sel=True, maxEta=2.1):
    mask = events_in["tau_pt"]>minPt
    if eta_sel:
        mask = mask & (events_in["tau_eta"]<maxEta) & (events_in["tau_eta"]>-maxEta)
    return mask

def deepTau_selection(events_in, deepTau_thr):
    true_taus_pred = events_in["deepTau_VSjet"] # deepTau prediction for tau vs jets
    mask = true_taus_pred >= deepTau_thr
    return mask

def iso_tau_selection(events_in, var_abs, var_rel):
    iso_cut = (events_in[var_abs]>0) | (events_in[var_rel]>0)
    return iso_cut

def ditau_selection(taus_mask):
    ev_mask = taus_mask.sum()>=2
    # events_out = events_in[ev_mask]
    # return events_out, ev_mask
    return ev_mask

