import ROOT
from ROOT import TTree
import numpy as np
import uproot
import awkward as ak
from selection import *

class dataset():

    def __init__(self, fileName, treeName, treeName_gen, is_MC=True, is_old=False):
        self.fileName = fileName
        self.treeName = treeName
        self.treeName_gen = treeName_gen
        self.is_old  = is_old
        self.is_MC = is_MC

    def get_events(self):
        tree_in = uproot.open(self.fileName)[self.treeName] 
        events = ak.Table(tree_in.arrays(namedecode="utf-8")) # filtered events
        return events
    
    def get_gen_events(self):
        tree_gen = uproot.open(self.fileName)[self.treeName_gen]
        events = ak.Table(tree_gen.arrays(namedecode="utf-8")) # generator events
        return events
    
    def get_taus(self, apply_selection=True):
        events = self.get_events()
        if self.is_old:
            taus = ak.JaggedArray.zip(e=events.tau_e, pt=events.tau_pt, eta=events.tau_eta, phi=events.tau_phi, 
                                    looseIsoAbs=events.tau_looseIsoAbs, looseIsoRel=events.tau_looseIsoRel,
                                    mediumIsoAbs=events.tau_mediumIsoAbs, mediumIsoRel=events.tau_mediumIsoRel,
                                    tightIsoAbs=events.tau_tightIsoAbs, tightIsoRel=events.tau_tightIsoRel,
                                    gen_e = events.gen_tau_e, gen_pt=events.gen_tau_pt, gen_eta=events.gen_tau_eta, gen_phi=events.gen_tau_phi,
                                    lepton_gen_match=events.lepton_gen_match, deepTau_VSjet=events.deepTau_VSjet)
        else:
            taus = ak.JaggedArray.zip(e=events.tau_e, pt=events.tau_pt, eta=events.tau_eta, phi=events.tau_phi, 
                                    looseIsoAbs=events.tau_looseIsoAbs, looseIsoRel=events.tau_looseIsoRel,
                                    mediumIsoAbs=events.tau_mediumIsoAbs, mediumIsoRel=events.tau_mediumIsoRel,
                                    tightIsoAbs=events.tau_tightIsoAbs, tightIsoRel=events.tau_tightIsoRel,
                                    gen_e = events.gen_tau_e, gen_pt=events.gen_tau_pt, gen_eta=events.gen_tau_eta, gen_phi=events.gen_tau_phi,
                                    lepton_gen_match=events.lepton_gen_match, deepTau_VSjet=events.deepTau_VSjet, vz=events.tau_vz)
            if apply_selection:
                L1taus = ak.JaggedArray.zip(e = events.L1tau_e, pt=events.L1tau_pt, eta=events.L1tau_eta, phi=events.L1tau_phi)
                taus = L1THLTTauMatching(L1taus, taus)
                taus = HLTJetPairDzMatchFilter(taus)
        
        return taus

    def get_gen_taus(self):
        events = self.get_gen_events()
        gen_taus = ak.JaggedArray.zip(gen_e = events.gen_tau_e, gen_pt=events.gen_tau_pt, gen_eta=events.gen_tau_eta, gen_phi=events.gen_tau_phi,
                                    lepton_gen_match=events.lepton_gen_match)
        return gen_taus