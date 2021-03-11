from eff_rate import *
from dataset import dataset
import json
import ROOT
from ROOT import *
from selection import *
ROOT.gROOT.SetBatch(True)

data_path = '/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/'
fileName = "VBFToTauTau_minPt20_pre10setup.root"
QCD_fileJson = "QCD_samples.json"

Pt_thr_list = [20, 25, 30, 35, 40, 45]

# get VBF sample
treeName_gen = "gen_counter"
treeName_in = "initial_counter"
dataset_eff = dataset(data_path + fileName, treeName_in, treeName_gen)
taus = dataset_eff.get_taus()
gen_taus = dataset_eff.get_gen_taus()

# get QCD sample
print("Getting QCD samples")
QCD_taus_list = []
QCD_xs_list = []
QCD_den_list = []
with open(QCD_fileJson, "r") as json_file:
    samples = json.load(json_file)
    for key, value in samples.items():
        data = dataset(data_path + value[0], treeName_in, treeName_gen)
        QCD_taus_list.append(data.get_taus())
        QCD_xs_list.append(value[1])
        QCD_den_list.append(len(data.get_gen_events()))
# print(QCD_xs_list)

thr_list = np.flip(np.linspace(0.0, 1.0, num=101))
eff_atThreshold = []

# with PdfPages(plot_path + 'eff_vs_pt_{}.pdf'.format(plot_name)) as pdf:

for n, Pt_thr in enumerate(Pt_thr_list):
    print("\nComputing rate improvement at {} GeV".format(Pt_thr))

    # Compute cut-based medium efficiency and rate
    eff_base, _, _ = compute_isocut_eff(taus, gen_taus, "mediumIsoAbs", "mediumIsoRel", Pt_thr=Pt_thr)
    rate_base = 0
    for i, QCD_taus in enumerate(QCD_taus_list):
        rate_i, _, _ = compute_isocut_rate(QCD_taus, QCD_den_list[i], "mediumIsoAbs", "mediumIsoRel", Pt_thr=Pt_thr, is_MC=True, xs=QCD_xs_list[i])
        # print(rate_i)
        rate_base = rate_base + rate_i
    
    # Find deepTau threshold such that efficiency is closest to cut-based efficiency
    eff_list = []
    pos = -1
    for i, thr in enumerate(thr_list):
        eff, _, _ = compute_deepTau_eff(taus, gen_taus, thr, Pt_thr=Pt_thr)
        eff_list.append(eff)
        if eff > eff_base:
            pos = i
            print(eff_list[pos])
            print(eff_list[i-1])
            print(eff_base)
            break
    if pos == -1:
        print("All efficiency below threshold")
        sys.exit(1)

    rate = 0
    for j, QCD_taus in enumerate(QCD_taus_list):
        rate_j, _, _ = compute_deepTau_rate(
            QCD_taus, QCD_den_list[j], thr_list[pos], Pt_thr=Pt_thr, is_MC=True, xs=QCD_xs_list[j])
        rate = rate + rate_j
    print(rate)
    print(rate_base)


    

    