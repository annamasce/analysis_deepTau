import numpy as np
import awkward as ak


def delta_r2(v1, v2):
    """
    Calculates deltaR squared between two particles v1, v2 whose
    eta and phi methods return arrays
    """
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi ** 2 + deta ** 2
    return dr2


def delta_r(v1, v2):
    """
    Calculates deltaR between two particles v1, v2 whose
    eta and phi methods return arrays.
    
    Note: Prefer delta_r2 for cuts.
    """
    return np.sqrt(delta_r2(v1, v2))


def delta_z(v1, v2):
    """
    Calculates dz between two particles v1, v2 whose
    vz methods return arrays.
    """
    dz = v1.vz - v2.vz
    return dz

def deep_thr_lin1(tau, par, Pt_thr):
    a_1 = (par[1] - par[0]) / (100 - Pt_thr)
    b_1 = par[1] - 100 * a_1
    c = 0.125
    a_2 = (c - par[1]) / 200
    b_2 = c - 300 * a_2

    thr1 = ak.where(tau.pt < 100, a_1 * tau.pt + b_1, 0)
    thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
    thr3 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2 + thr3
    return deep_thr

def deep_thr_lin1_lowThr(tau, par, Pt_thr):
    a_1 = (par[1] - par[0]) / (100 - Pt_thr)
    b_1 = par[1] - 100 * a_1
    c = 0.05
    a_2 = (c - par[1]) / 200
    b_2 = c - 300 * a_2

    thr1 = ak.where(tau.pt < 100, a_1 * tau.pt + b_1, 0)
    thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
    thr3 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2 + thr3
    return deep_thr


def deep_thr_parab(tau, par, Pt_thr):
    a_1 = (par[1] - par[0]) / (-10000 - Pt_thr ** 2 + 200 * Pt_thr)
    b_1 = -200 * a_1
    c_1 = par[1] - a_1 * 100 ** 2 - b_1 * 100

    c = 0.125
    a_2 = (c - par[1]) / 200
    b_2 = c - 300 * a_2

    thr1 = ak.where(tau.pt < 100, a_1 * tau.pt ** 2 + b_1 * tau.pt + c_1, 0)
    thr2 = ak.where((tau.pt >= 100) & (tau.pt < 300), a_2 * tau.pt + b_2, 0)
    thr3 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2 + thr3
    return deep_thr


def deep_thr_lin2(tau, par, Pt_thr):
    c = 0.125
    m = (c - par[0]) / (300 - Pt_thr)
    q = c - m * 300
    thr1 = ak.where((tau.pt < 300), m * tau.pt + q, 0)
    thr2 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2
    return deep_thr

def deep_thr_lin2_tauMET(tau, par, Pt_thr):
    c = 0.8
    m = (c - par[0]) / (300 - Pt_thr)
    q = c - m * 300
    thr1 = ak.where((tau.pt < 300), m * tau.pt + q, 0)
    thr2 = ak.where(tau.pt >= 300, c, 0)
    deep_thr = thr1 + thr2
    return deep_thr

def deep_thr_lin2_highPt(tau, par, Pt_thr, Pt_step=300):
    c = 0.125
    m = (c - par[0]) / (Pt_step - Pt_thr)
    q = c - m * Pt_step
    thr1 = ak.where((tau.pt < Pt_step), m * tau.pt + q, 0)
    thr2 = ak.where(tau.pt >= Pt_step, c, 0)
    deep_thr = thr1 + thr2
    return deep_thr

def deep_thr_lin1_tauMET(tau, par, Pt_thr):
    a_1 = (par[1] - par[0]) / (300 - Pt_thr)
    b_1 = par[1] - 300 * a_1
    c = 0.5
    a_2 = (c - par[1]) / 200
    b_2 = c - 500 * a_2

    thr1 = ak.where(tau.pt < 300, a_1 * tau.pt + b_1, 0)
    thr2 = ak.where((tau.pt >= 300) & (tau.pt < 500), a_2 * tau.pt + b_2, 0)
    thr3 = ak.where(tau.pt >= 500, c, 0)
    deep_thr = thr1 + thr2 + thr3
    return deep_thr

def deep_thr_const(tau, par, Pt_thr):
    c = par[0]
    thr = ak.where(tau.pt > Pt_thr, c, 0)
    return thr


def true_tau_selection(taus):
    tau_mask = taus.lepton_gen_match == 5
    return tau_mask


def gen_tau_selection(taus, gen_minPt=20, gen_maxEta=2.1):
    tau_mask = (taus.gen_pt > gen_minPt) & (abs(taus.gen_eta) < gen_maxEta)
    return tau_mask


def reco_tau_selection(taus, minPt=20, eta_sel=True, maxEta=2.1):
    tau_mask = taus.pt > minPt
    if eta_sel:
        tau_mask_final = tau_mask & (abs(taus.eta) < maxEta)
        return tau_mask_final
    return tau_mask


def deepTau_selection(taus, deepTau_thr):
    true_taus_pred = taus.deepTau_VSjet  # deepTau prediction for tau vs jets
    tau_mask = (true_taus_pred >= deepTau_thr)
    return tau_mask

def deepTau_selection_ptdep(taus, Pt_thr, par, deep_thr):
    true_taus_pred = taus.deepTau_VSjet  # deepTau prediction for tau vs jets
    tau_mask = (true_taus_pred >= deep_thr(taus, par, Pt_thr))
    return tau_mask


def iso_tau_selection(taus, var_abs, var_rel):
    tau_mask = (taus[var_abs] > 0) | (taus[var_rel] > 0)
    return tau_mask


def ditau_selection(mask_tau_1, mask_tau_2):
    # require at least one pair of good taus per event
    ev_mask = ak.sum(mask_tau_1 & mask_tau_2, axis=-1) >= 1
    return ev_mask

def L1seed_correction(L1taus, taus):
    L1taus_mask = (L1taus.pt >= 32)
    ev_mask = ak.sum(L1taus_mask, axis=1) >= 2
    return (L1taus[L1taus_mask])[ev_mask], taus[ev_mask]
    # return L1taus[ev_mask], taus[ev_mask]

def L1THLTTauMatching(L1taus, taus):
    dR_matching = 0.5
    tau_inpair, L1_inpair = ak.unzip(ak.cartesian([taus, L1taus], nested=True))
    # dR = delta_r(L1_inpair, tau_inpair)
    dR = delta_r(tau_inpair, L1_inpair)
    # # print(dR[range(2)])
    # # consider only L1taus for which there is at least 1 matched tau
    # tau_inpair = tau_inpair[dR<dR_matching]
    mask = ak.sum(dR < dR_matching, axis=-1) > 0
    # tau_inpair = tau_inpair[mask]
    # # take first matched tau for each L1tau
    # L2taus = tau_inpair[:,:,0]
    # # print(L2taus[range(2)])
    L2taus = taus[mask]
    return L2taus


def HLTJetPairDzMatchFilter(L2taus):
    jetMinPt = 20.0
    jetMaxEta = 2.1
    jetMinDR = 0.5
    jetMaxDZ = 0.2
    L2taus = L2taus[reco_tau_selection(L2taus, minPt=jetMinPt, maxEta=jetMaxEta)]
    # Take all possible pairs of L2 taus
    L2tau_1, L2tau_2 = ak.unzip(ak.combinations(L2taus, 2, axis=1))
    dr2 = delta_r2(L2tau_1, L2tau_2)
    dz = delta_z(L2tau_1, L2tau_2)
    pair_mask = (dr2 >= jetMinDR * jetMinDR) & (abs(dz) <= jetMaxDZ)
    # ev_mask = ak.sum(pair_mask, axis=1) > 0

    return L2tau_1[pair_mask], L2tau_2[pair_mask]

def DzMatchFilter(tau_1, tau_2):
    jetMaxDZ = 0.2
    jetMinDR = 0.5
    mask_1 = tau_1.passed_last_filter > 0
    mask_2 = tau_2.passed_last_filter > 0
    pair_mask = mask_1 & mask_2
    dr2 = delta_r2(tau_1, tau_2)
    dz = delta_z(tau_1, tau_2)
    pair_mask = pair_mask & (dr2 >= jetMinDR * jetMinDR) & (abs(dz) <= jetMaxDZ)

    return tau_1[pair_mask], tau_2[pair_mask]

def true_other_selection(gen_part):
    mask = (gen_part.lepton_gen_match == 2) | (gen_part.lepton_gen_match == 4)
    return mask

def gen_other_selection(gen_part, gen_minPt=24, gen_maxEta=2.1):
    mask = (gen_part.gen_pt > gen_minPt) & (abs(gen_part.gen_eta) < gen_maxEta)
    return mask

def evt_base_selection(dataset):
    gen_events = dataset.get_gen_events()
    gen_leptons = dataset.get_gen_taus()
    tau_mask = true_tau_selection(gen_leptons) & gen_tau_selection(gen_leptons)
    other_mask = true_other_selection(gen_leptons) & gen_other_selection(gen_leptons)
    ev_mask = (ak.sum(tau_mask, axis=-1) > 0) & (ak.sum(other_mask, axis=-1) > 0)
    good_events = gen_events[ev_mask].evt
    return ak.to_numpy(good_events, False)

def good_evt_selection(events, good_events):
    evt_ids = ak.to_numpy(events.evt, False)
    matches = np.isin(evt_ids, good_events)
    # matches = [True if ev in good_events else False for ev in evt_ids]
    return matches