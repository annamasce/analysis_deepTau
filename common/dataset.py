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

l2_thr = {
    "DiTau": 0.386,
    # "DiTau": 0.339,
    # "HighPtTau": 0.1809,
    "HighPtTau": 0.035,
    # "TauMET": 0.5940
    "TauMET": 0.9535
}

class Dataset:
    def __init__(self, fileName, treeName, treeName_gen, type="DiTau", apply_l2=False, decay_modes=False):
        self.fileName = fileName
        self.treeName = treeName
        self.treeName_gen = treeName_gen
        self.type = type
        self.apply_l2 = apply_l2
        self.decay_modes = decay_modes

    @staticmethod
    def compute_decay_mode(nChargedHad, nNeutralHad):
        return (nChargedHad - 1) * 5 + nNeutralHad

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
        # Apply L2 filter for concerned paths - NOTE: L2 score could be set to -1 in ntuples for studies
        if (self.type in ["HighPtTau", "TauMET"]) and self.apply_l2:
            print("applying l2 filter")
            l2_mask = (events.l2nn_output >= l2_thr[self.type]) | (events.l1_pt >= 250)
            ev_mask = ak.sum(l2_mask, axis=-1) >= 1
            events = events[ev_mask]
        if (self.type == "DiTau") and self.apply_l2:
            print("applying l1 filter")
            # Apply also L1 filter to Di-Tau path to study different L1 thresholds
            l1_mask = (events.l1_pt >= 32)
            ev_mask = ak.sum(l1_mask, axis=-1) >= 2
            events = events[ev_mask]
            print("applying l2 filter")
            l2_mask = (events.l2nn_output >= l2_thr[self.type]) | (events.l1_pt >= 250)
            ev_mask = ak.sum(l2_mask, axis=-1) >= 2
            events = events[ev_mask]
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
                     "deepTau_VSjet": events.deepTau_VSjet, "deepTau_VSe": events.deepTau_VSe,
                     "passed_last_filter": events.tau_passedLastFilter, "vz": events.tau_vz}
        if self.decay_modes:
            taus_dict["gen_decay_mode"] = Dataset.compute_decay_mode(events.gen_tau_nChargedHadrons, events.gen_tau_nNeutralHadrons)
        taus = ak.zip(taus_dict)
        index = ak.argsort(taus.pt, ascending=False)
        taus = taus[index]
        return taus

    def get_taupairs(self, apply_selection=True):
        events = self.get_events()
        taus = self.get_taus()
        tau_1, tau_2 = ak.unzip(ak.combinations(taus, 2, axis=1))
        # Necessary selection for old ntuples production (last filters to be applied offline)
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
        gen_taus_dict = {"gen_e": events.gen_tau_e, "gen_pt": events.gen_tau_pt, "gen_eta": events.gen_tau_eta,
                           "gen_phi": events.gen_tau_phi,
                           "lepton_gen_match": events.lepton_gen_match}
        if self.decay_modes:
            gen_taus_dict["gen_decay_mode"] = Dataset.compute_decay_mode(events.gen_tau_nChargedHadrons, events.gen_tau_nNeutralHadrons)
        gen_taus = ak.zip(gen_taus_dict)
        index = ak.argsort(gen_taus.gen_pt, ascending=False)
        gen_taus = gen_taus[index]
        return gen_taus

    def get_gen_taupairs(self):
        gen_taus = self.get_gen_taus()
        gen_tau_1, gen_tau_2 = ak.unzip(ak.combinations(gen_taus, 2, axis=1))
        return gen_tau_1, gen_tau_2

    def get_met(self):
        events = self.get_events()
        MET_dict = {"e": events.MET_e, "pt": events.MET_pt, "eta": events.MET_eta, "phi": events.MET_phi}
        METs = ak.zip(MET_dict)
        index = ak.argsort(METs.pt, ascending=False)
        METs = METs[index]
        return METs

    def get_gen_met(self):
        events = self.get_gen_events()
        gen_met_dict = {"gen_pt": events.gen_met_calo_pt, "gen_phi": events.gen_met_calo_phi}
        gen_met = ak.zip(gen_met_dict)
        return gen_met

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
        elif self.type == "HighPtTau":
            ev_mask = (ak.sum(tau_mask, axis=-1) == 1)
        elif self.type == "TauMET":
            gen_met = self.get_gen_met()
            gen_met_mask = (gen_met.gen_pt > 100)
            ev_mask = (ak.sum(tau_mask, axis=-1) == 1) & (ak.sum(gen_met_mask, axis=-1) > 0)
        elif self.type == "DiTau":
            ev_mask = (ak.sum(tau_mask, axis=-1) > 1)
        else:
            sys.exit("Wrong dataset type. choose one of the following: EleTau, MuTau, TauMET, HighPtTau, DiTau")
        good_events = gen_events[ev_mask].evt
        good_events = ak.to_numpy(good_events, False)
        print(good_events)
        return ev_mask, good_events


