from common.selection import *
from functools import partial

paths = {
    "DiTau": "HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4",
    "EleTau": "HLT_Ele24_eta2p1_WPTight_Gsf_MediumChargedIsoPFTauHPS30_eta2p1_CrossL1_v1",
    "MuTau": "HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1_v4",
    "TauMET": "HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100_v12",
    "HighPtTau": "HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_v12"
    # "HighPtTau": "HLT_MediumChargedIsoPFTau180HighPtRelaxedIso_Trk50_eta2p1_1pr_v11"
}

# optim_pars_paths = {
#     "DiTau": [deep_thr_lin1, [0.57085716, 0.37163761]],
#     "EleTau": [deep_thr_lin1, [0.78537964, 0.5286338]],
#     # "EleTau": [deep_thr_lin1, [0.77, 0.6]],
#     # "MuTau": [deep_thr_lin1, [0.46743955, 0.3611882]],
#     "MuTau": [deep_thr_lin1_lowThr, [0.48133268, 0.31975578]],
#     # "TauMET": [deep_thr_lin1_tauMET, [0.97836487, 0.90928622]],
#     "TauMET": [deep_thr_lin1_tauMET, [0.96245395, 0.945]],
#     "HighPtTau": [partial(deep_thr_lin2_highPt, Pt_step=500), [0.65320599]]
# }

optim_pars_paths = {
    # "DiTau": [deep_thr_lin1_lowThr, [0.6065, 0.4384]],
    "DiTau": [deep_thr_lin1_lowThr, [0.61898598, 0.64095102]],
    # "EleTau": [deep_thr_lin1_lowThr, [0.7045, 0.7029]],
    "EleTau": [deep_thr_lin1_lowThr, [0.3980, 0.6668, 0.4844]],
    "MuTau": [deep_thr_lin1_lowThr, [0.5419, 0.4837]],
    # "TauMET": [deep_thr_lin1_tauMET, [0.958, 0.919]],
    "TauMET": [deep_thr_lin1_tauMET, [0.9619, 0.9269]],
    # "HighPtTau": [partial(deep_thr_lin2_highPt, Pt_step=500), [0.6072]]
    "HighPtTau": [partial(deep_thr_lin2_highPt, Pt_step=500), [0.7421]]
}

rate_bm_paths = {
    "DiTau": 30.4,
    "EleTau": 8.62,
    "MuTau": 4.47,
    "TauMET": 1.92,
    "HighPtTau": 12.77
}

Pt_thr_paths = {
    "DiTau": 35,
    "EleTau": 30,
    "MuTau": 27,
    "TauMET": 50,
    "HighPtTau": 180
}