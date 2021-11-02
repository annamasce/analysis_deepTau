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
    def __init__(self, fileName, treeName, treeName_gen):
        self.fileName = fileName
        self.treeName = treeName
        self.treeName_gen = treeName_gen
        events = uproot.lazy(self.__define_tree_expression(is_gen=False))
        self.__events = events
        gen_events = uproot.lazy(self.__define_tree_expression(is_gen=True))
        self.__gen_events = gen_events

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
        return self.__events

    def get_gen_events(self):
        return self.__gen_events

    def get_taus(self, events=None):
        if events is None:
            events = self.__events
        taus_dict = {"e": events.tau_e, "pt": events.tau_pt, "eta": events.tau_eta, "phi": events.tau_phi,
                     "looseIsoAbs": events.tau_looseIsoAbs, "looseIsoRel": events.tau_looseIsoRel,
                     "mediumIsoAbs": events.tau_mediumIsoAbs, "mediumIsoRel": events.tau_mediumIsoRel,
                     "tightIsoAbs": events.tau_tightIsoAbs, "tightIsoRel": events.tau_tightIsoRel,
                     "gen_e": events.gen_tau_e, "gen_pt": events.gen_tau_pt, "gen_eta": events.gen_tau_eta,
                     "gen_phi": events.gen_tau_phi,
                     "lepton_gen_match": events.lepton_gen_match, "deepTau_VSjet": events.deepTau_VSjet,
                     "vz": events.tau_vz,
                     "idx": ak.local_index(events.tau_pt, axis=-1)}
        taus = ak.zip(taus_dict)
        return taus

    def get_L1taus(self, events=None):
        if events is None:
            events = self.__events
        L1taus = ak.zip({"e": events.L1tau_e, "pt": events.L1tau_pt, "eta": events.L1tau_eta,
                         "phi": events.L1tau_phi})
        return L1taus

    def get_tau_pairs(self, apply_selection=True):
        events = self.get_events()
        taus = self.get_taus()
        # index = ak.argsort(taus.pt, ascending=False)
        # taus = taus[index]
        # tau_1, tau_2 = ak.unzip(ak.combinations(taus, 2, axis=1))
        if apply_selection:
            L1taus = self.get_L1taus()
            tau_1, tau_2 = apply_ditauHLT_selection(events, taus, L1taus, return_events=False)
            return tau_1, tau_2
        index = ak.argsort(taus.pt, ascending=False)
        taus = taus[index]
        tau_1, tau_2 = ak.unzip(ak.combinations(taus, 2, axis=1))
        return tau_1, tau_2

    def get_gen_taus(self):
        events = self.get_gen_events()
        gen_taus = ak.zip({"gen_e": events.gen_tau_e, "gen_pt": events.gen_tau_pt, "gen_eta": events.gen_tau_eta,
                           "gen_phi": events.gen_tau_phi,
                           "lepton_gen_match": events.lepton_gen_match})
        return gen_taus

    def get_gen_tau_pairs(self):
        gen_taus = self.get_gen_taus()
        index = ak.argsort(gen_taus.gen_pt, ascending=False)
        gen_taus = gen_taus[index]
        gen_tau_1, gen_tau_2 = ak.unzip(ak.combinations(gen_taus, 2, axis=1))
        return gen_tau_1, gen_tau_2


# class DatasetOffline(Dataset):
#     def get_off_taus(self, events=None):
#         if events is None:
#             events = self.get_events()
#         off_taus_dict = {"e": events.off_tau_e, "pt": events.off_tau_pt, "eta": events.off_tau_eta,
#                          "phi": events.off_tau_phi,
#                          "gen_e": events.off_gen_tau_e, "gen_pt": events.off_gen_tau_pt,
#                          "gen_eta": events.off_gen_tau_eta,
#                          "gen_phi": events.off_gen_tau_phi,
#                          "lepton_gen_match": events.off_lepton_gen_match, "deepTau_VSjet": events.off_deepTau_VSjet,
#                          "deepTau_VSe": events.off_deepTau_VSe, "deepTau_VSmu": events.off_deepTau_VSmu,
#                          "dz": events.off_tau_dz, "decayModeFinding": events.off_decayModeFinding,
#                          "decayMode": events.off_decayMode,
#                          "idx": ak.local_index(events.off_tau_pt, axis=-1)}
#         off_taus = ak.zip(off_taus_dict)
#         return off_taus

class DatasetOffline():

    def __init__(self, fileName, treeName):
        self.fileName = fileName
        self.treeName = treeName
        events = uproot.lazy(self.__define_tree_expression(is_gen=False))
        self.__events = events

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
        return self.__events

    def get_off_taus(self, events=None):
        if events is None:
            events = self.get_events()
        off_taus_dict = {"e": events.tau_off_e, "pt": events.tau_off_pt, "eta": events.tau_off_eta,
                         "phi": events.tau_off_phi,
                         "gen_e": events.gen_tau_off_e, "gen_pt": events.gen_tau_off_pt,
                         "gen_eta": events.gen_tau_off_eta,
                         "gen_phi": events.gen_tau_off_phi,
                         "lepton_gen_match": events.tau_off_lepton_gen_match, "deepTau_VSjet": events.tau_off_deepTau_VSjet,
                         "deepTau_VSe": events.tau_off_deepTau_VSe, "deepTau_VSmu": events.tau_off_deepTau_VSmu,
                         "dz": events.tau_off_dz, "decayModeFinding": events.tau_off_decayModeFinding,
                         "decayMode": events.tau_off_decayMode,
                         "idx": ak.local_index(events.tau_off_pt, axis=-1)}
        off_taus = ak.zip(off_taus_dict)
        return off_taus

    def get_trig_taus(self, events=None):
        if events is None:
            events = self.get_events()
        trig_taus_dict = {"e": events.tau_trig_e, "pt": events.tau_trig_pt, "eta": events.tau_trig_eta,
                         "phi": events.tau_trig_phi,
                         "idx": ak.local_index(events.tau_trig_pt, axis=-1)}
        trig_taus = ak.zip(trig_taus_dict)
        return trig_taus
