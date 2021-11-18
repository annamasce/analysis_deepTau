import uproot
import sys
import awkward as ak
from common.selection import *

import collections
import six

def iterable(arg):
    return (
        isinstance(arg, collections.abc.Iterable)
        and not isinstance(arg, six.string_types)
    )

class Dataset:
    def __init__(self, fileName, treeName, treeName_gen, type="DiTau"):
        self.fileName = fileName
        self.treeName = treeName
        self.treeName_gen = treeName_gen
        self.type = type

    def __define_tree_expression(self, is_gen):
        if is_gen:
            treeName = self.treeName_gen
        else:
            treeName = self.treeName
        if iterable(self.fileName):
            tree_path = []
            for file in self.fileName:
                tree_path.append(file + ":" + treeName)
        else:
            tree_path = self.fileName + ":" + treeName
        return tree_path


    def get_events(self):
        # tree_in = uproot.open(self.fileName)[self.treeName]
        # events = tree_in.arrays()  # filtered events
        tree_path = self.__define_tree_expression(is_gen=False)
        events = uproot.lazy(tree_path)
        return events

    def get_gen_events(self):
        # tree_gen = uproot.open(self.fileName)[self.treeName_gen]
        # events = tree_gen.arrays()  # generator events
        tree_path = self.__define_tree_expression(is_gen=True)
        events = uproot.lazy(tree_path)
        return events

    def get_taus(self):
        events = self.get_events()
        taus_dict = {"e": events.tau_e, "pt": events.tau_pt, "eta": events.tau_eta, "phi": events.tau_phi,
                     "gen_e": events.gen_tau_e, "gen_pt": events.gen_tau_pt, "gen_eta": events.gen_tau_eta,
                     "gen_phi": events.gen_tau_phi, "lepton_gen_match": events.lepton_gen_match,
                     "deepTau_VSjet": events.deepTau_VSjet, "passed_last_filter": events.tau_passedLastFilter,
                     "vz": events.tau_vz}
        taus = ak.zip(taus_dict)
        index = ak.argsort(taus.pt, ascending=False)
        taus = taus[index]
        return taus

    def get_taupairs(self, apply_selection=True):
        events = self.get_events()
        taus = self.get_taus()
        tau_1, tau_2 = ak.unzip(ak.combinations(taus, 2, axis=1))
        if apply_selection:
            L1taus = ak.zip({"e": events.L1tau_e, "pt": events.L1tau_pt, "eta": events.L1tau_eta,
                             "phi": events.L1tau_phi})
            # apply L1seed correction in case Pt28 and Pt30 seeds are considered
            L1taus, taus = L1seed_correction(L1taus, taus)
            # match taus with L1 taus
            taus = L1THLTTauMatching(L1taus, taus)
            tau_1, tau_2 = HLTJetPairDzMatchFilter(taus)
        # Return all possible pairs of tau which pass preselection
        return tau_1, tau_2

    def get_gen_taus(self):
        events = self.get_gen_events()
        gen_taus = ak.zip({"gen_e": events.gen_tau_e, "gen_pt": events.gen_tau_pt, "gen_eta": events.gen_tau_eta,
                           "gen_phi": events.gen_tau_phi,
                           "lepton_gen_match": events.lepton_gen_match})
        index = ak.argsort(gen_taus.gen_pt, ascending=False)
        gen_taus = gen_taus[index]
        return gen_taus

    def get_gen_taupairs(self):
        gen_taus = self.get_gen_taus()
        gen_tau_1, gen_tau_2 = ak.unzip(ak.combinations(gen_taus, 2, axis=1))
        return gen_tau_1, gen_tau_2

    def get_MET(self):
        events = self.get_events()
        MET_dict = {"e": events.MET_e, "pt": events.MET_pt, "eta": events.MET_eta, "phi": events.MET_phi}
        METs = ak.zip(MET_dict)
        index = ak.argsort(METs.pt, ascending=False)
        METs = METs[index]
        return METs

    def evt_base_selection(self):
        gen_events = self.get_gen_events()
        gen_leptons = self.get_gen_taus()
        tau_mask = true_tau_selection(gen_leptons) & gen_tau_selection(gen_leptons)
        if self.type == "EleTau":
            truth_mask = (gen_leptons.lepton_gen_match == 1) | (gen_leptons.lepton_gen_match == 3)
            gen_mask = (gen_leptons.gen_pt > 24) & (abs(gen_leptons.gen_eta) < 2.1)
            other_mask = truth_mask & gen_mask
            ev_mask = (ak.sum(tau_mask, axis=-1) > 0) & (ak.sum(other_mask, axis=-1) > 0)
        elif self.type == "MuTau":
            truth_mask = (gen_leptons.lepton_gen_match == 2) | (gen_leptons.lepton_gen_match == 4)
            gen_mask = (gen_leptons.gen_pt > 20) & (abs(gen_leptons.gen_eta) < 2.1)
            other_mask = truth_mask & gen_mask
            ev_mask = (ak.sum(tau_mask, axis=-1) > 0) & (ak.sum(other_mask, axis=-1) > 0)
        elif self.type in ["TauMET", "HighPtTau"]:
            ev_mask = (ak.sum(tau_mask, axis=-1) == 1)
        elif self.type == "DiTau":
            ev_mask = (ak.sum(tau_mask, axis=-1) > 1)
        else:
            sys.exit("Wrong dataset type. choose one of the following: EleTau, MuTau, TauMET, HighPtTau, DiTau")
        good_events = gen_events[ev_mask].evt
        print(good_events)
        return ev_mask, good_events


