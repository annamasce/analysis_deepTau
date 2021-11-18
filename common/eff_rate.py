from common.selection import *
from statsmodels.stats.proportion import proportion_confint

lumi = 122.792 / 7319  # Recorded luminsoity divided by delta_t for run 325022 LS=[64,377]
# lumi = 2e-2
# L1rate = 75817.94

############## EFFICIENCY ###########################

def num_mask_eff(taus, Pt_thr=20):
    num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr,
                                                                                           eta_sel=False)
    return num_tau_mask

def den_mask_eff(gen_taus):
    den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
    return den_tau_mask

def apply_numden_masks(tau_1, tau_2, gen_tau_1, gen_tau_2, Pt_thr=20):
    num_tau_mask_1 = num_mask_eff(tau_1, Pt_thr=Pt_thr)
    num_tau_mask_2 = num_mask_eff(tau_2, Pt_thr=Pt_thr)
    den_tau_mask_1 = den_mask_eff(gen_tau_1)
    den_tau_mask_2 = den_mask_eff(gen_tau_2)
    return num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2

def compute_eff_witherr(Nev_num, Nev_den):
    conf_int = proportion_confint(count=Nev_num, nobs=Nev_den, alpha=0.32, method='beta')
    eff = Nev_num / Nev_den
    err_low = eff - conf_int[0]
    err_up = conf_int[1] - eff
    return eff, err_low, err_up

def compute_base_eff_diTau(tau_1, tau_2, gen_tau_1, gen_tau_2, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path by applying only generator and true tau selection
    """
    num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(tau_1, tau_2, gen_tau_1, gen_tau_2, Pt_thr=Pt_thr)
    Nev_num = ak.sum(ditau_selection(num_tau_mask_1, num_tau_mask_2))
    Nev_den = ak.sum(ditau_selection(den_tau_mask_1, den_tau_mask_2))
    eff = Nev_num / Nev_den
    return eff

def compute_base_eff_singleTau(dataset, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path by applying only generator and true tau selection
    """
    good_ev_mask, good_events = dataset.evt_base_selection()
    taus = dataset.get_taus()
    gen_taus = dataset.get_gen_taus()
    num_tau_mask = (taus.passed_last_filter > 0) & num_mask_eff(taus, Pt_thr=Pt_thr)
    den_tau_mask = den_mask_eff(gen_taus)
    num_evt_mask = (ak.sum(num_tau_mask, axis=-1) > 0) & good_evt_selection(dataset.get_events(), good_events)
    den_evt_mask = (ak.sum(den_tau_mask, axis=-1) > 0) & good_ev_mask
    Nev_num = ak.sum(num_evt_mask)
    print(Nev_num)
    Nev_den = ak.sum(den_evt_mask)
    print(Nev_den)
    eff, err_low, err_up = compute_eff_witherr(Nev_num, Nev_den)
    return eff, err_low, err_up


def compute_deepTau_eff_diTau(tau_1, tau_2, gen_tau_1, gen_tau_2, deepTau_thr, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path at a certain deepTau threshold
    """
    num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(tau_1, tau_2, gen_tau_1, gen_tau_2, Pt_thr=Pt_thr)
    num_tau_mask_final_1 = deepTau_selection(tau_1, deepTau_thr) & num_tau_mask_1
    num_tau_mask_final_2 = deepTau_selection(tau_2, deepTau_thr) & num_tau_mask_2
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
    Nev_den = ak.sum(ditau_selection(den_tau_mask_1, den_tau_mask_2))
    eff, err_low, err_up = compute_eff_witherr(Nev_num, Nev_den)

    return eff, err_low, err_up

def compute_deepTau_ptdep_eff(tau_1, tau_2, gen_tau_1, gen_tau_2, par, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path at a pt dependent deepTau threshold
    """
    num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(tau_1, tau_2, gen_tau_1, gen_tau_2, Pt_thr=Pt_thr)
    num_tau_mask_final_1 = deepTau_selection_ptdep(tau_1, Pt_thr, par) & num_tau_mask_1
    num_tau_mask_final_2 = deepTau_selection_ptdep(tau_2, Pt_thr, par) & num_tau_mask_2
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
    Nev_den = ak.sum(ditau_selection(den_tau_mask_1, den_tau_mask_2))
    eff, err_low, err_up = compute_eff_witherr(Nev_num, Nev_den)

    return eff, err_low, err_up


def compute_deepTau_eff_list_diTau(tau_1, tau_2, gen_tau_1, gen_tau_2, thr_list, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path at different deepTau thresholds (given in thr_list)
    """
    num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(tau_1, tau_2, gen_tau_1, gen_tau_2, Pt_thr=Pt_thr)
    Nev_den = ak.sum(ditau_selection(den_tau_mask_1, den_tau_mask_2))

    eff_list = []
    err_list_low = []
    err_list_up = []

    for thr in thr_list:
        num_tau_mask_final_1 = deepTau_selection(tau_1, thr) & num_tau_mask_1
        num_tau_mask_final_2 = deepTau_selection(tau_2, thr) & num_tau_mask_2
        Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
        eff, err_low, err_up = compute_eff_witherr(Nev_num, Nev_den)
        eff_list.append(eff)
        err_list_low.append(err_low)
        err_list_up.append(err_up)

    return eff_list, err_list_low, err_list_up

def compute_deepTau_eff_list_singleTau(dataset, thr_list, Pt_thr=20):
    """
    Computes efficiency of single-tau HLT path at different deepTau thresholds (given in thr_list)
    """
    good_ev_mask, good_events = dataset.evt_base_selection()
    taus = dataset.get_taus()
    gen_taus = dataset.get_gen_taus()
    num_tau_mask = (taus.passed_last_filter > 0) & num_mask_eff(taus, Pt_thr=Pt_thr)
    den_tau_mask = den_mask_eff(gen_taus)
    num_evt_mask = (ak.sum(num_tau_mask, axis=-1) > 0) & good_evt_selection(dataset.get_events(), good_events)
    den_evt_mask = (ak.sum(den_tau_mask, axis=-1) > 0) & good_ev_mask
    Nev_den = ak.sum(den_evt_mask)

    eff_list = []
    err_list_low = []
    err_list_up = []

    for thr in thr_list:
        num_tau_mask_final = deepTau_selection(taus, thr) & num_tau_mask
        num_evt_mask_final = num_evt_mask & ak.sum(num_tau_mask_final, axis=-1) > 0
        Nev_num = ak.sum(num_evt_mask_final)
        eff, err_low, err_up = compute_eff_witherr(Nev_num, Nev_den)
        eff_list.append(eff)
        err_list_low.append(err_low)
        err_list_up.append(err_up)

    return eff_list, err_list_low, err_list_up



def compute_isocut_eff(tau_1, tau_2, gen_tau_1, gen_tau_2, var_abs, var_rel, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path using isolation cut
    """
    num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(tau_1, tau_2, gen_tau_1,
                                                                                        gen_tau_2, Pt_thr=Pt_thr)
    iso_cut_1 = (tau_1[var_abs] > 0) | (tau_1[var_rel] > 0)
    iso_cut_2 = (tau_2[var_abs] > 0) | (tau_2[var_rel] > 0)
    num_tau_mask_final_1 = num_tau_mask_1 & iso_cut_1
    num_tau_mask_final_2 = num_tau_mask_2 & iso_cut_2
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
    Nev_den = ak.sum(ditau_selection(den_tau_mask_1, den_tau_mask_2))

    eff, err_low, err_up = compute_eff_witherr(Nev_num, Nev_den)

    return eff, err_low, err_up

############## RATE ###########################

def compute_rate_witherr(Nev_num, Nev_den, is_MC=False, L1rate=75817.94, xs=469700.0):

    conf_int = proportion_confint(count=Nev_num, nobs=Nev_den, alpha=0.32, method='beta')

    if is_MC:
        rate = Nev_num * xs * lumi / Nev_den
        err_low = rate - conf_int[0] * xs * lumi
        err_up = conf_int[1] * xs * lumi - rate
    else:
        rate = Nev_num * L1rate / Nev_den
        err_low = rate - conf_int[0] * L1rate
        err_up = conf_int[1] * L1rate - rate

    return rate, err_low, err_up


def compute_deepTau_rate_diTau(tau_1, tau_2, Nev_den, deepTau_thr, Pt_thr=20, L1rate=75817.94, is_MC=False, xs=469700.0):
    """
    Computes rate of di-tau HLT path at a certain deepTau threshold 
    """

    num_tau_mask_1 = reco_tau_selection(tau_1, minPt=Pt_thr)
    num_tau_mask_2 = reco_tau_selection(tau_2, minPt=Pt_thr)

    num_tau_mask_final_1 = deepTau_selection(tau_1, deepTau_thr) & num_tau_mask_1
    num_tau_mask_final_2 = deepTau_selection(tau_2, deepTau_thr) & num_tau_mask_2
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
    rate, err_low, err_up = compute_rate_witherr(Nev_num, Nev_den, is_MC=is_MC, L1rate=L1rate, xs=xs)

    return rate, err_low, err_up

def compute_deepTau_rate_singleTau(taus, Nev_den, deepTau_thr, Pt_thr=20, L1rate=75817.94):
    """
    Computes rate of di-tau HLT path at a certain deepTau threshold
    """
    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr) & deepTau_selection(taus, deepTau_thr)
    Nev_num = ak.sum(ak.sum(num_tau_mask, axis=-1) > 0)
    rate, err_low, err_up = compute_rate_witherr(Nev_num, Nev_den, L1rate=L1rate)

    return rate, err_low, err_up

def compute_deepTau_ptdep_rate_diTau(tau_1, tau_2, Nev_den, par, Pt_thr=20, L1rate=75817.94, is_MC=False, xs=469700.0):
    """
    Computes rate of di-tau HLT path for a pt dependent deepTau thr
    """

    num_tau_mask_1 = reco_tau_selection(tau_1, minPt=Pt_thr)
    num_tau_mask_2 = reco_tau_selection(tau_2, minPt=Pt_thr)

    num_tau_mask_final_1 = deepTau_selection_ptdep(tau_1, Pt_thr, par) & num_tau_mask_1
    num_tau_mask_final_2 = deepTau_selection_ptdep(tau_2, Pt_thr, par) & num_tau_mask_2
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
    rate, err_low, err_up = compute_rate_witherr(Nev_num, Nev_den, is_MC=is_MC, L1rate=L1rate, xs=xs)

    return rate, err_low, err_up

def compute_deepTau_ptdep_rate_singleTau(taus, Nev_den, par, Pt_thr=20, L1rate=75817.94):
    """
    Computes rate of single-tau HLT path for a pt dependent deepTau thr
    """

    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr) & deepTau_selection_ptdep(taus, Pt_thr, par)
    Nev_num = ak.sum(ak.sum(num_tau_mask, axis=-1) > 0)
    rate, err_low, err_up = compute_rate_witherr(Nev_num, Nev_den, L1rate=L1rate)

    return rate, err_low, err_up


def compute_deepTau_rate_list_diTau(tau_1, tau_2, Nev_den, thr_list, Pt_thr=20, is_MC=False, L1rate=75817.94, xs=469700.0):
    '''
    Computes rate of di-tau HLT path at different deepTau thresholds (given in thr_list) 
    '''
    rate_list = []
    err_list_low = []
    err_list_up = []
    for thr in thr_list:
        rate_results = compute_deepTau_rate_diTau(tau_1, tau_2, Nev_den, thr, Pt_thr=Pt_thr, is_MC=is_MC, L1rate=L1rate, xs=xs)
        rate_list.append(rate_results[0])
        err_list_low.append(rate_results[1])
        err_list_up.append(rate_results[2])

    return rate_list, err_list_low, err_list_up

def compute_deepTau_rate_list_singleTau(taus, Nev_den, thr_list, Pt_thr=20, L1rate=75817.94):
    '''
    Computes rate of single-tau HLT path at different deepTau thresholds (given in thr_list)
    '''
    rate_list = []
    err_list_low = []
    err_list_up = []
    for thr in thr_list:
        rate_results = compute_deepTau_rate_singleTau(taus, Nev_den, thr, Pt_thr=Pt_thr, L1rate=L1rate)
        rate_list.append(rate_results[0])
        err_list_low.append(rate_results[1])
        err_list_up.append(rate_results[2])

    return rate_list, err_list_low, err_list_up


def compute_isocut_rate(tau_1, tau_2, Nev_den, var_abs, var_rel, Pt_thr=20, is_MC=False, L1rate=75817.94, xs=469700.0):
    """
    Computes rate of di-tau HLT path using isolation cut
    """
    num_tau_mask_1 = reco_tau_selection(tau_1, minPt=Pt_thr)
    iso_cut_1 = (tau_1[var_abs] > 0) | (tau_1[var_rel] > 0)
    num_tau_mask_2 = reco_tau_selection(tau_2, minPt=Pt_thr)
    iso_cut_2 = (tau_2[var_abs] > 0) | (tau_2[var_rel] > 0)
    num_tau_mask_final_1 = num_tau_mask_1 & iso_cut_1
    num_tau_mask_final_2 = num_tau_mask_2 & iso_cut_2
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final_1, num_tau_mask_final_2))
    rate, err_low, err_up = compute_rate_witherr(Nev_num, Nev_den, is_MC=is_MC, L1rate=L1rate, xs=xs)

    return rate, err_low, err_up

def compute_base_rate_singleTau(taus, Nev_den, Pt_thr=20, L1rate=75817.94):
    """
    Computes rate of single-tau HLT path by applying only reco-tau selection
    """
    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr)
    Nev_num = ak.sum(ak.sum(num_tau_mask, axis=-1) > 0)
    rate, err_low, err_up = compute_rate_witherr(Nev_num, Nev_den, L1rate=L1rate)

    return rate, err_low, err_up

