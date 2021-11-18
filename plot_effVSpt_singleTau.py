import sys
import ROOT
from ROOT import *
from array import array
import argparse
from common.eff_rate import *
from common.dataset import Dataset
from HLT_paths import paths, optim_pars_paths, Pt_thr_paths


def set_eff1Dhist_style(hist, Pt_thr, Pt_max):
    hist.GetXaxis().SetTitle("gen p_{T} tau [GeV]")
    hist.GetYaxis().SetTitle("Efficiency")
    hist.GetXaxis().SetMoreLogLabels(kTRUE)
    hist.GetYaxis().SetTitleOffset(1.2)
    hist.GetXaxis().SetTitleOffset(1.2)
    hist.GetXaxis().SetRangeUser(Pt_thr-5., Pt_max)
    hist.SetMarkerSize(1.5)
    return hist

if __name__ == '__main__':
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2()

    parser = argparse.ArgumentParser()
    parser.add_argument("plotName", help="name of the pdf plot")
    parser.add_argument("datasetType", help="dataset type to identify proper generator selection for efficiency")
    args = parser.parse_args()
    if args.datasetType not in ["EleTau", "MuTau", "DiTau"]:
        sys.exit("Wrong dataset type. choose one of the following: EleTau, MuTau, TauMET, HighPtTau, DiTau")

    plot_name = args.plotName
    plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/"
    data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/211109/"
    fileName_eff = data_path + "VBFHToTauTau_deepTau.root"
    fileName_eff_base = data_path + "VBFHToTauTau_oldHLT.root"
    fileName_rates = data_path + "Ephemeral_deepTau.root"
    treeName_gen = "gen_counter"
    treeName_in = "final_{}_counter".format(args.datasetType)
    treeName_in_base = "final_{}_counter".format(paths[args.datasetType])

    # L1 rate
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792 / 7319
    L1rate_bm = L1rate * lumi_bm / lumi_real

    # get taus for efficiency
    print("Loading sample for efficiency")
    dataset_eff = Dataset(fileName_eff, treeName_in, treeName_gen, type=args.datasetType)
    taus = dataset_eff.get_taus()
    gen_taus = dataset_eff.get_gen_taus()

    # get taus for old HLT efficiency
    print("Loading sample for old HLT efficiency")
    dataset_eff_base = Dataset(fileName_eff_base, treeName_in_base, treeName_gen, type=args.datasetType)
    taus_base = dataset_eff_base.get_taus()
    gen_taus_base = dataset_eff_base.get_gen_taus()

    # get taus for rates
    print("Loading sample for rate")
    dataset_rates = Dataset(fileName_rates, treeName_in, treeName_gen, type=args.datasetType)
    taus_rates = dataset_rates.get_taus()
    taus_rates = taus_rates[taus_rates.passed_last_filter > 0]
    Nev_den = len(dataset_rates.get_gen_events())

    Pt_thr = Pt_thr_paths[args.datasetType]
    Pt_bins = [Pt_thr, 35, 40, 45, 50, 60, 70, 100, 200, 500]
    nbins = len(Pt_bins) - 1

    print("Plotting differential efficiency vs gen Pt")
    par = optim_pars_paths[args.datasetType]
    good_ev_mask, good_events = dataset_eff.evt_base_selection()
    num_tau_mask = (taus.passed_last_filter > 0) & num_mask_eff(taus, Pt_thr=Pt_thr) & deepTau_selection_ptdep(taus, Pt_thr, par)
    den_tau_mask = den_mask_eff(gen_taus)
    if args.datasetType in ["TauMET", "HighPtTau"]:
        num_ev_mask = (ak.sum(num_tau_mask, axis=-1) == 1) & good_evt_selection(dataset_eff.get_events(), good_events)
        den_ev_mask = (ak.sum(den_tau_mask, axis=-1) == 1) & good_ev_mask
    else:
        num_ev_mask = (ak.sum(num_tau_mask, axis=-1) > 0) & good_evt_selection(dataset_eff.get_events(), good_events)
        den_ev_mask = (ak.sum(den_tau_mask, axis=-1) > 0) & good_ev_mask
    tau_num = ak.firsts(taus[num_tau_mask][num_ev_mask], axis=-1)
    tau_den = ak.firsts(gen_taus[den_tau_mask][den_ev_mask], axis=-1)

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
    good_ev_mask_base, good_events_base = dataset_eff_base.evt_base_selection()
    num_tau_mask_base = (taus_base.passed_last_filter > 0) & num_mask_eff(taus_base, Pt_thr)
    den_tau_mask_base = den_mask_eff(gen_taus_base)
    num_ev_mask_base = (ak.sum(num_tau_mask_base, axis=-1) > 0) & good_evt_selection(dataset_eff_base.get_events(), good_events_base)
    den_ev_mask_base = (ak.sum(den_tau_mask_base, axis=-1) > 0) & good_ev_mask_base
    tau_num_base = ak.firsts(taus_base[num_tau_mask_base][num_ev_mask_base], axis=-1)
    tau_den_base = ak.firsts(gen_taus_base[den_tau_mask_base][den_ev_mask_base], axis=-1)

    num_hist_1D_base = TH1D("num_1d_base", "", nbins, array("d", Pt_bins))
    den_hist_1D_base = TH1D("den_1d_base", "", nbins, array("d", Pt_bins))

    for i in range(len(tau_num_base)):
        num_hist_1D_base.Fill(tau_num_base.gen_pt[i])
    for i in range(len(tau_den_base)):
        den_hist_1D_base.Fill(tau_den_base.gen_pt[i])

    eff_hist_1D_base = num_hist_1D_base.Clone("eff_1d_base")
    eff_hist_1D_base.Divide(den_hist_1D_base)

    # eff_hist_1D_base = set_eff1Dhist_style(eff_hist_1D_base, Pt_thr, Pt_bins[-1], cut_based=True)

    # Plotting
    gStyle.SetOptStat(0)
    drawCanv_1d = TCanvas("c1", "")
    gPad.SetLogx()
    gStyle.SetPaintTextFormat("1.2f")
    eff_hist_1D.Draw()
    eff_hist_1D_base.SetLineColor(2)
    eff_hist_1D_base.Draw("same")
    legend = TLegend(0.6, 0.1, 0.9, 0.3)
    legend.AddEntry(eff_hist_1D, "DeepTau benchmark")
    legend.AddEntry(eff_hist_1D_base, "Run 2 WP")
    legend.Draw()
    drawCanv_1d.Print(plot_path + args.datasetType + "/diffeff_VSgenPt1D_" + plot_name + ".pdf")


    # print("Total efficiency:")
    # eff_deep = compute_deepTau_ptdep_eff(taus[0], taus[1], gen_taus[0], gen_taus[1], par, Pt_thr=Pt_thr)
    # print(eff_deep)
    print("Total rate:")
    rate_deep = compute_deepTau_ptdep_rate_singleTau(taus_rates, Nev_den, par, Pt_thr=Pt_thr,
                                           L1rate=L1rate_bm)
    print(rate_deep)

