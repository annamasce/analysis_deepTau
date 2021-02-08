import sys
sys.path.append('../')
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

def compute_eff(taus, gen_taus, thr_list, Pt_thr=20):
    '''
    Computes efficiency of di-tau HLT path at different deepTau thresholds (given in thr_list)
    '''
    Nev_num_list = []
    num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr, eta_sel=False)
    den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
    Nev_den = ditau_selection(den_tau_mask).sum()

    for thr in thr_list:
        num_tau_mask_final = deepTau_selection(taus, thr) & num_tau_mask
        Nev_num = ditau_selection(num_tau_mask_final).sum()
        Nev_num_list.append(Nev_num)
            
    eff_list = np.array(Nev_num_list)/Nev_den
    
    return eff_list

def compute_rates(taus, Nev_den, thr_list, Pt_thr=20):
    '''
    Computes rate of di-tau HLT path at different deepTau thresholds (given in thr_list)
    '''
    Nev_num_list = []
    L1rate = 73455.34
    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr)

    for thr in thr_list:
        num_tau_mask_final = deepTau_selection(taus, thr) & num_tau_mask
        Nev_num = ditau_selection(num_tau_mask_final).sum()
        Nev_num_list.append(Nev_num)

    Nev_num_list = np.array(Nev_num_list)
    rates = Nev_num_list*L1rate/Nev_den
    
    return rates

def compute_isocut_eff(taus, gen_taus, var_abs, var_rel, Pt_thr=20):
    '''
    Computes efficiency of di-tau HLT path using isolation cut
    '''
    num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr, eta_sel=False)
    den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
    Nev_den = ditau_selection(den_tau_mask).sum()
    iso_cut = (taus[var_abs]>0) | (taus[var_rel]>0)
    num_tau_mask_final = num_tau_mask & iso_cut
    Nev_num = ditau_selection(num_tau_mask_final).sum()
    eff = Nev_num/Nev_den

    return eff

def compute_isocut_rate(taus, Nev_den, var_abs, var_rel, Pt_thr=20):
    '''
    Computes rate of di-tau HLT path using isolation cut
    '''
    L1rate = 73455.34
    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr)
    iso_cut = (taus[var_abs]>0) | (taus[var_rel]>0)
    num_tau_mask_final = num_tau_mask & iso_cut
    Nev_num = ditau_selection(num_tau_mask_final).sum()
    rate = Nev_num*L1rate/Nev_den
    return rate