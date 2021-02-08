import ROOT
from ROOT import TTree
from sklearn.metrics import roc_curve, auc
import numpy as np
import uproot
import awkward as ak
import sys
from helpers import *

def true_tau_selection(taus):
    mask = taus.lepton_gen_match==5
    return mask

def gen_tau_selection(taus, gen_minPt=20, gen_maxEta=2.1):
    mask = (taus.gen_pt>gen_minPt) & (taus.gen_eta<gen_maxEta) & (taus.gen_eta> -gen_maxEta)
    return mask

def reco_tau_selection(taus, minPt=20, eta_sel=True, maxEta=2.1):
    mask = taus.pt>minPt
    if eta_sel:
        mask_final = mask & (taus.eta<maxEta) & (taus.eta>-maxEta)
        return mask_final
    return mask

def deepTau_selection(taus, deepTau_thr):
    true_taus_pred = taus.deepTau_VSjet # deepTau prediction for tau vs jets
    mask = (true_taus_pred >= deepTau_thr)
    return mask

def iso_tau_selection(taus, var_abs, var_rel):
    iso_cut = (taus[var_abs]>0) | (taus[var_rel]>0)
    return iso_cut

def ditau_selection(taus_mask):
    ev_mask = taus_mask.sum()>=2
    # events_out = events_in[ev_mask]
    # return events_out, ev_mask
    return ev_mask

def L1THLTTauMatching(L1taus, taus):
    dR_matching = 0.5
    # take all possible pairs of L1taus and taus
    L1_inpair, tau_inpair = L1taus.cross(taus, nested=True).unzip()
    dR = delta_r(L1_inpair, tau_inpair)
    # print(dR[range(2)])
    # consider only L1taus for which there is at least 1 matched tau
    tau_inpair = tau_inpair[dR<dR_matching]
    mask = (dR<dR_matching).sum()>0
    tau_inpair = tau_inpair[mask]
    # take first matched tau for each L1tau
    L2taus = tau_inpair[:,:,0]
    # print(L2taus[range(2)])
    return L2taus

def HLTJetPairDzMatchFilter(L2taus):
    jetMinPt = 20.0
    jetMaxEta = 2.1
    jetMinDR = 0.5
    jetMaxDZ = 0.2
    L2taus = L2taus[reco_tau_selection(L2taus, minPt=jetMinPt, maxEta=jetMaxEta)]
    # Take all possible pairs of L2 taus
    L2tau_1, L2tau_2 = L2taus.distincts().unzip()
    dr2 = delta_r2(L2tau_1, L2tau_2)
    # print(dr2[range(5)])
    dz = delta_z(L2tau_1, L2tau_2)
    # print(dz[range(5)])
    pair_mask = (dr2 >= jetMinDR*jetMinDR) & (abs(dz) <= jetMaxDZ)
    ev_mask = pair_mask.sum()>0 
    # return events[ev_mask], L2taus[ev_mask].compact()
    # print(L2taus[ev_mask].compact()[range(5)])
    return L2taus[ev_mask].compact()



