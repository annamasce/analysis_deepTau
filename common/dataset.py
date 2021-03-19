import uproot
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
    def __init__(self, fileName, treeName, treeName_gen, is_MC=True, is_old=False):
        self.fileName = fileName
        self.treeName = treeName
        self.treeName_gen = treeName_gen
        self.is_old = is_old
        self.is_MC = is_MC

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

    def get_taus(self, apply_selection=True):
        events = self.get_events()
        taus_dict = {"e": events.tau_e, "pt": events.tau_pt, "eta": events.tau_eta, "phi": events.tau_phi,
                     "looseIsoAbs": events.tau_looseIsoAbs, "looseIsoRel": events.tau_looseIsoRel,
                     "mediumIsoAbs": events.tau_mediumIsoAbs, "mediumIsoRel": events.tau_mediumIsoRel,
                     "tightIsoAbs": events.tau_tightIsoAbs, "tightIsoRel": events.tau_tightIsoRel,
                     "gen_e": events.gen_tau_e, "gen_pt": events.gen_tau_pt, "gen_eta": events.gen_tau_eta,
                     "gen_phi": events.gen_tau_phi,
                     "lepton_gen_match": events.lepton_gen_match, "deepTau_VSjet": events.deepTau_VSjet}
        if self.is_old:
            taus = ak.zip(taus_dict)
            index = ak.argsort(taus.pt, ascending=False)
            taus = taus[index]
            tau_1, tau_2 = ak.unzip(ak.combinations(taus, 2, axis=1))
        else:
            taus_dict["vz"] = events.tau_vz
            taus = ak.zip(taus_dict)
            index = ak.argsort(taus.pt, ascending=False)
            taus = taus[index]
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
        gen_tau_1, gen_tau_2 = ak.unzip(ak.combinations(gen_taus, 2, axis=1))
        return gen_tau_1, gen_tau_2
