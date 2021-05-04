import math
from scipy import optimize
from common.eff_rate import *
from common.dataset import Dataset

data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/"
fileName_eff = "VBFHToTauTau.root"
fileName_rates = "EphemeralHLTPhysics_1to8.root"

Pt_thr_list = [20, 25, 30, 35, 40, 45]

# get VBF sample
treeName_gen = "gen_counter"
treeName_in = "final_counter"
dataset_eff = Dataset(data_path + fileName_eff, treeName_in, treeName_gen)
taus = dataset_eff.get_tau_pairs()
gen_taus = dataset_eff.get_gen_tau_pairs()

# get HLT physics sample
dataset_rates = Dataset(data_path + fileName_rates, treeName_in, treeName_gen)
taus_rates = dataset_rates.get_tau_pairs()
Nev_den = len(dataset_rates.get_gen_events())

for n, Pt_thr in enumerate(Pt_thr_list):
    print("\nComputing rate improvement at {} GeV".format(Pt_thr))
    eff_base, _, _ = compute_isocut_eff(taus[0], taus[1], gen_taus[0], gen_taus[1], "mediumIsoAbs", "mediumIsoRel", Pt_thr=Pt_thr)
    print(eff_base)

    def f(thr):
        eff, _, _ = compute_deepTau_eff(taus[0], taus[1], gen_taus[0], gen_taus[1], thr, Pt_thr=Pt_thr)
        return eff - eff_base

    solution = optimize.root_scalar(f, bracket=[0, 1], method='bisect')
    deep_thr = solution.root
    rate_base = compute_isocut_rate(taus_rates[0], taus_rates[1], Nev_den, "mediumIsoAbs", "mediumIsoRel", Pt_thr=Pt_thr)
    rate = compute_deepTau_rate(taus_rates[0], taus_rates[1], Nev_den, deep_thr, Pt_thr=Pt_thr)

    print(rate_base[0])
    print(rate[0])
    diff = rate_base[0] - rate[0]
    diff_err = math.sqrt(rate_base[1]**2 + rate[1]**2)
    print("diff", diff, "+/-", diff_err)
    print("perc", diff/rate_base[0])


    

    