import sys
import ROOT
from ROOT import *
from array import array
import argparse
from common.eff_rate import *
from common.dataset import Dataset
from HLT_paths import paths, optim_pars_paths, Pt_thr_paths
import matplotlib.pyplot as plt

DM = None


def set_eff1Dhist_style(hist, Pt_thr, Pt_max):
    hist.GetXaxis().SetTitle("gen p_{T} tau [GeV]")
    hist.GetYaxis().SetTitle("Efficiency")
    hist.GetXaxis().SetMoreLogLabels(kTRUE)
    hist.GetYaxis().SetTitleOffset(1.2)
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.GetYaxis().SetRangeUser(0.01, 1.01)
    hist.SetMarkerSize(0.7)
    hist.SetMarkerStyle(kFullCircle)
    return hist

def taus_preselected(datasets, Pt_thr, decay_mode=None):
    # select only true taus that pass generator preselection
    num_taus_list = []
    den_taus_list = []
    for dataset in datasets:
        good_ev_mask, good_events = dataset.evt_base_selection()
        print(good_events)
        taus = dataset.get_taus()
        gen_taus = dataset.get_gen_taus()
        num_tau_mask = (taus.passed_last_filter > 0) & num_mask_eff(taus, Pt_thr=Pt_thr)
        den_tau_mask = den_mask_eff(gen_taus)
        if decay_mode is not None:
            num_tau_mask = num_tau_mask & (taus.gen_decay_mode == decay_mode)
            den_tau_mask = den_tau_mask & (gen_taus.gen_decay_mode == decay_mode)
        if dataset.type in ["TauMET", "HighPtTau"]:
            num_ev_mask = (ak.sum(num_tau_mask, axis=-1) == 1) & good_evt_selection(dataset.get_events(),
                                                                                    good_events)
            den_ev_mask = (ak.sum(den_tau_mask, axis=-1) == 1) & good_ev_mask
        else:
            num_ev_mask = (ak.sum(num_tau_mask, axis=-1) > 0) & good_evt_selection(dataset.get_events(),
                                                                                   good_events)
            den_ev_mask = (ak.sum(den_tau_mask, axis=-1) > 0) & good_ev_mask
        num_taus_selected = (taus[num_tau_mask])[num_ev_mask]
        den_taus_selected = (gen_taus[den_tau_mask][den_ev_mask])
        num_taus_list.append(num_taus_selected)
        den_taus_list.append(den_taus_selected)
    num_taus_array_final = ak.concatenate(num_taus_list)
    den_taus_array_final = ak.concatenate(den_taus_list)
    return num_taus_array_final, den_taus_array_final

if __name__ == '__main__':
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2()

    parser = argparse.ArgumentParser()
    parser.add_argument("plotName", help="name of the pdf plot")
    parser.add_argument("datasetType", help="dataset type to identify proper generator selection for efficiency")
    parser.add_argument("--remove_l2", help="remove L2 filter", action="store_true")
    parser.add_argument("--optimise_VSe", help="optimise deepTau VSe thr", action="store_true")
    args = parser.parse_args()
    apply_l2 = not args.remove_l2
    if args.datasetType not in ["EleTau", "MuTau", "DiTau", "TauMET", "HighPtTau"]:
        sys.exit("Wrong dataset type. choose one of the following: EleTau, MuTau, TauMET, HighPtTau, DiTau")

    plot_name = args.plotName
    plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/"
    data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/"
    fileName_rates = data_path + "220409/Ephemeral_deepTau.root"
    treeName_gen = "gen_counter"
    treeName_in = "final_{}_counter".format(args.datasetType)
    treeName_in_base = "final_{}_counter".format(paths[args.datasetType])

    # Create root file to save efficiency histograms
    hfile = TFile('{}/{}/RootFiles/histos_{}.root'.format(plot_path, args.datasetType, plot_name), 'RECREATE', 'ROOT file with histograms')

    # L1 rate
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792 / 7319
    L1rate_bm = L1rate * lumi_bm / lumi_real

    # get taus for efficiency
    print("Loading sample for efficiency")
    datasets_eff = [Dataset(data_path + "220330/ZprimeToTauTau_deepTau.root", treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2, decay_modes=True), Dataset(data_path + "220330/VBFHToTauTau_deepTau.root", treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2, decay_modes=True)]
    # datasets_eff = [
    #     # Dataset(data_path + "211213/ZprimeToTauTau_deepTau.root", treeName_in, treeName_gen, type=args.datasetType,
    #     #         apply_l2=apply_l2),
    #     # Dataset(data_path + "211213/VBFHToTauTau_deepTau.root", treeName_in, treeName_gen, type=args.datasetType,
    #     #         apply_l2=apply_l2),
    #     Dataset(data_path + "211213/WjetsToLNu_deepTau.root", treeName_in, treeName_gen, type=args.datasetType,
    #             apply_l2=apply_l2)]

    # get taus for old HLT efficiency
    print("Loading sample for old HLT efficiency")
    datasets_eff_base = [Dataset(data_path + "220405/ZprimeToTauTau_oldHLT.root", treeName_in_base, treeName_gen, type=args.datasetType, decay_modes=False), Dataset(data_path + "220405/VBFHToTauTau_oldHLT.root", treeName_in_base, treeName_gen, type=args.datasetType, decay_modes=False)]
    # datasets_eff_base = [Dataset(data_path + "211109/WjetsToLNu_oldHLT.root", treeName_in_base, treeName_gen, type=args.datasetType)]

    # get taus for rates
    print("Loading sample for rate")
    dataset_rates = Dataset(fileName_rates, treeName_in, treeName_gen, type=args.datasetType, apply_l2=apply_l2)
    taus_rates = dataset_rates.get_taus()
    taus_rates = taus_rates[taus_rates.passed_last_filter > 0]
    Nev_den = len(dataset_rates.get_gen_events())

    par = optim_pars_paths[args.datasetType][1]
    deep_thr = optim_pars_paths[args.datasetType][0]
    Pt_thr = Pt_thr_paths[args.datasetType]
    Pt_bins = [Pt_thr, 35, 40, 45, 50, 60, 70, 100, 200, 500, 700, 1000, 2000]
    # Pt_bins = [Pt_thr, 200, 250, 300, 400, 500, 700, 1000, 2000, 5000]
    # Pt_bins = [Pt_thr, 60, 70, 100, 150, 300]
    nbins = len(Pt_bins) - 1

    print("Plotting differential efficiency vs gen Pt")
    taus_num, taus_den = taus_preselected(datasets_eff, Pt_thr, decay_mode=DM)
    if args.optimise_VSe:
        deepTau_mask = deepTau_selection_ptdep(taus_num, Pt_thr, par[1:], deep_thr) & (taus_num.deepTau_VSe > deep_thr_VSele(taus_num, par[0]))
    else:
        deepTau_mask = deepTau_selection_ptdep(taus_num, Pt_thr, par, deep_thr)
    taus_num = (taus_num[deepTau_mask])[ak.sum(deepTau_mask, axis=-1) > 0]
    tau_num = ak.firsts(taus_num)
    tau_den = ak.firsts(taus_den)

    # numerator histogram
    num_hist_1D = TH1D("num_1d", "", nbins, array("d", Pt_bins))
    # denominator histogram
    den_hist_1D = TH1D("den_1d", "", nbins, array("d", Pt_bins))

    # Fill histograms with gen tau pt of leading taus
    for i in range(len(tau_num)):
        num_hist_1D.Fill(tau_num.gen_pt[i])
    for i in range(len(tau_den)):
        den_hist_1D.Fill(tau_den.gen_pt[i])

    # Compute efficiency
    eff_hist_1D = num_hist_1D.Clone("eff_1d")
    eff_hist_1D.Divide(den_hist_1D)

    eff_hist_1D = set_eff1Dhist_style(eff_hist_1D, Pt_thr, Pt_bins[-1])

    # Compute Run 2 WP efficiency
    print("Plotting old HLT differential efficiency vs gen Pt")
    taus_num_base, taus_den_base = taus_preselected(datasets_eff_base, Pt_thr, decay_mode=DM)
    tau_num_base = ak.firsts(taus_num_base)
    tau_den_base = ak.firsts(taus_den_base)

    num_hist_1D_base = TH1D("num_1d_base", "", nbins, array("d", Pt_bins))
    den_hist_1D_base = TH1D("den_1d_base", "", nbins, array("d", Pt_bins))

    for i in range(len(tau_num_base)):
        num_hist_1D_base.Fill(tau_num_base.gen_pt[i])
    for i in range(len(tau_den_base)):
        den_hist_1D_base.Fill(tau_den_base.gen_pt[i])

    eff_hist_1D_base = num_hist_1D_base.Clone("eff_1d_base")
    eff_hist_1D_base.Divide(den_hist_1D_base)

    eff_hist_1D_base = set_eff1Dhist_style(eff_hist_1D_base, Pt_thr, Pt_bins[-1])

    # Plotting
    gStyle.SetOptStat(0)
    drawCanv_1d = TCanvas("c1", "")
    gPad.SetLogx()
    gStyle.SetPaintTextFormat("1.2f")
    eff_hist_1D.SetMarkerColor(4)
    eff_hist_1D.SetLineColor(4)
    eff_hist_1D.Draw("PE1")
    eff_hist_1D_base.SetMarkerColor(2)
    eff_hist_1D_base.SetLineColor(2)
    eff_hist_1D_base.Draw("PE1 same")
    legend = TLegend(0.55, 0.15, 0.85, 0.3)
    # legend = TLegend(0.15, 0.7, 0.4, 0.85)
    legend.AddEntry(eff_hist_1D, "DeepTau benchmark")
    legend.AddEntry(eff_hist_1D_base, "Run 2 WP")
    legend.Draw()
    drawCanv_1d.Print(plot_path + args.datasetType + "/diffeff_VSgenPt1D_" + plot_name + ".pdf")

    hfile.Write()

    print("Total rate:")
    rate_deep = compute_deepTau_ptdep_rate_singleTau(taus_rates, Nev_den, par, deep_thr, Pt_thr=Pt_thr,
                                           L1rate=L1rate_bm, optimise_VSe=args.optimise_VSe)
    print(rate_deep)

    plt.plot(tau_num.pt, deep_thr(tau_num, par, Pt_thr), ".")
    plt.xlim(0, 500)
    plt.show()
