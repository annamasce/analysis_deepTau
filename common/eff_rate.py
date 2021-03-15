from common.selection import *
from statsmodels.stats.proportion import proportion_confint

lumi = 122.792 / 7319  # Recorded luminsoity divided by delta_t for run 325022 LS=[64,377]
# lumi = 2e-2
L1rate = 75817.94

# def num_den_mask_eff(taus, gen_taus, Pt_thr=20):
#     num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr,
#                                                                                            eta_sel=False)
#     den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
#
#     return num_tau_mask, den_tau_mask

def compute_base_eff(taus, gen_taus, Pt_thr=20):
    '''
    Computes efficiency of di-tau HLT path at by applying only generator and true tau selection
    '''
    num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr,
                                                                                           eta_sel=False)
    den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
    Nev_num = ak.sum(ditau_selection(num_tau_mask))
    Nev_den = ak.sum(ditau_selection(den_tau_mask))
    eff = Nev_num / Nev_den
    return eff


def compute_deepTau_eff(taus, gen_taus, deepTau_thr, Pt_thr=20):
    '''
    Computes efficiency of di-tau HLT path at a certain deepTau threshold
    '''
    num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr,
                                                                                           eta_sel=False)
    num_tau_mask_final = deepTau_selection(taus, deepTau_thr) & num_tau_mask
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final))
    den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
    Nev_den = ak.sum(ditau_selection(den_tau_mask))

    conf_int = proportion_confint(count=Nev_num, nobs=Nev_den, alpha=0.32, method='beta')
    eff = Nev_num / Nev_den
    err_low = eff - conf_int[0]
    err_up = conf_int[1] - eff

    return eff, err_low, err_up


def compute_deepTau_eff_list(taus, gen_taus, thr_list, Pt_thr=20):
    '''
    Computes efficiency of di-tau HLT path at different deepTau thresholds (given in thr_list)
    '''
    eff_list = []
    err_list_low = []
    err_list_up = []
    for thr in thr_list:
        eff_results = compute_deepTau_eff(taus, gen_taus, thr, Pt_thr=Pt_thr)
        eff_list.append(eff_results[0])
        err_list_low.append(eff_results[1])
        err_list_up.append(eff_results[2])
    return eff_list, err_list_low, err_list_up


def compute_deepTau_rate(taus, Nev_den, deepTau_thr, Pt_thr=20, is_MC=False, xs=469700.0):
    '''
    Computes rate of di-tau HLT path at a certain deepTau threshold 
    '''
    # lumi = 432.27/37424.0 # Recorded luminsoity divided by delta_t for run 325022
    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr)
    num_tau_mask_final = deepTau_selection(taus, deepTau_thr) & num_tau_mask
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final))
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


def compute_deepTau_rate_list(taus, Nev_den, thr_list, Pt_thr=20, is_MC=False, xs=469700.0):
    '''
    Computes rate of di-tau HLT path at different deepTau thresholds (given in thr_list) 
    '''
    rate_list = []
    err_list_low = []
    err_list_up = []
    for thr in thr_list:
        rate_results = compute_deepTau_rate(taus, Nev_den, thr, Pt_thr=Pt_thr, is_MC=is_MC, xs=xs)
        rate_list.append(rate_results[0])
        err_list_low.append(rate_results[1])
        err_list_up.append(rate_results[2])

    return rate_list, err_list_low, err_list_up


def compute_isocut_eff(taus, gen_taus, var_abs, var_rel, Pt_thr=20):
    """
    Computes efficiency of di-tau HLT path using isolation cut
    """
    num_tau_mask = true_tau_selection(taus) & gen_tau_selection(taus) & reco_tau_selection(taus, minPt=Pt_thr,
                                                                                           eta_sel=False)
    den_tau_mask = true_tau_selection(gen_taus) & gen_tau_selection(gen_taus)
    Nev_den = ak.sum(ditau_selection(den_tau_mask))
    iso_cut = (taus[var_abs] > 0) | (taus[var_rel] > 0)
    num_tau_mask_final = num_tau_mask & iso_cut
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final))

    eff = Nev_num / Nev_den
    conf_int = proportion_confint(count=Nev_num, nobs=Nev_den, alpha=0.32, method='beta')
    err_low = eff - conf_int[0]
    err_up = conf_int[1] - eff

    return eff, err_low, err_up


def compute_isocut_rate(taus, Nev_den, var_abs, var_rel, Pt_thr=20, is_MC=False, xs=469700.0):
    """
    Computes rate of di-tau HLT path using isolation cut
    """
    # lumi = 432.27/37424.0 # Recorded luminsoity divided by delta_t for run 325022
    num_tau_mask = reco_tau_selection(taus, minPt=Pt_thr)
    iso_cut = (taus[var_abs] > 0) | (taus[var_rel] > 0)
    num_tau_mask_final = num_tau_mask & iso_cut
    Nev_num = ak.sum(ditau_selection(num_tau_mask_final))
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
