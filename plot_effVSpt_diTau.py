import json
import ROOT
from ROOT import *
from array import array
from scipy import optimize
import argparse
from common.selection import *
from common.eff_rate import *
from common.dataset import Dataset
from common.selection import DzMatchFilter
from HLT_paths import paths, optim_pars_paths, Pt_thr_paths


def set_eff2Dhist_style(hist, Pt_thr, Pt_max, cut_based = False):
    if cut_based:
        hist.SetTitle("Cut-based Medium Efficiency (Run 2 setup, p_{T} > %d GeV)" % Pt_thr)
    else:
        hist.SetTitle("Efficiency for p_{T} > %d GeV" % Pt_thr)
    hist.GetXaxis().SetTitle("gen p_{T} leading tau [GeV]")
    hist.GetYaxis().SetTitle("gen p_{T} subleading tau [GeV]")
    hist.GetXaxis().SetMoreLogLabels(kTRUE)
    hist.GetYaxis().SetMoreLogLabels(kTRUE)
    hist.GetYaxis().SetTitleOffset(1.3)
    hist.GetXaxis().SetTitleOffset(1.2)
    if Pt_thr==20:
        hist.GetXaxis().SetRangeUser(Pt_thr, Pt_max)
        hist.GetYaxis().SetRangeUser(Pt_thr, Pt_max)
    else:
        hist.GetXaxis().SetRangeUser(Pt_thr-5., Pt_max)
        hist.GetYaxis().SetRangeUser(Pt_thr-5., Pt_max)
    hist.SetMarkerSize(1.5)
    return hist

if __name__ == '__main__':
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.SetDefaultSumw2()

    parser = argparse.ArgumentParser()
    parser.add_argument("plotName", help="name of the pdf plot")
    parser.add_argument("--qcd", help="work on qcd input for rates", action="store_true")
    args = parser.parse_args()

    plot_name = args.plotName
    plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/DiTau/"
    data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/"
    fileName_eff_1 = "220330/VBFHToTauTau_deepTau.root"
    fileName_eff_2 = "220330/ZprimeToTauTau_deepTau.root"
    fileName_eff_base_1 = "220405/VBFHToTauTau_oldHLT.root"
    fileName_eff_base_2 = "220405/ZprimeToTauTau_oldHLT.root"
    fileName_rates = "220409/Ephemeral_deepTau.root"
    treeName_gen = "gen_counter"
    treeName_in = "final_DiTau_counter"
    treeName_in_base = "final_HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4_counter"

    # Create root file to save efficiency histograms
    hfile = TFile('{}RootFiles/histos_{}.root'.format(plot_path, plot_name), 'RECREATE',
                  'ROOT file with histograms')


    # L1 rate
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792 / 7319
    L1rate_bm = L1rate * lumi_bm / lumi_real

    # get taus for efficiency
    print("Loading sample for efficiency")
    files_eff = [data_path + fileName_eff_1, data_path + fileName_eff_2]
    dataset_eff = Dataset(files_eff, treeName_in, treeName_gen, type="DiTau", apply_l2=True)
    taus = dataset_eff.get_taupairs(apply_selection=False)
    taus = DzMatchFilter(taus[0], taus[1])
    gen_taus = dataset_eff.get_gen_taupairs()

    # get taus for old HLT efficiency
    print("Loading sample for old HLT efficiency")
    files_eff_base = [data_path + fileName_eff_base_1, data_path + fileName_eff_base_2]
    dataset_eff_base = Dataset(files_eff_base, treeName_in_base, treeName_gen, type="DiTau", apply_l2=False)
    taus_base = dataset_eff_base.get_taupairs(apply_selection=False)
    print(len(taus_base[0]))
    taus_base = DzMatchFilter(taus_base[0], taus_base[1])
    print(len(taus_base[0]))
    gen_taus_base = dataset_eff_base.get_gen_taupairs()
    print(len(gen_taus_base[0]))

    # get taus for rates
    print("Loading sample for rate")
    dataset_rates = Dataset(data_path + fileName_rates, treeName_in, treeName_gen, type="DiTau", apply_l2=True)
    taus_rates = dataset_rates.get_taupairs(apply_selection=False)
    taus_rates = DzMatchFilter(taus_rates[0], taus_rates[1])
    Nev_den = len(dataset_rates.get_gen_events())
    print(Nev_den)

    Pt_bins = [20, 25, 30, 35, 40, 45, 50, 60, 70, 100, 200, 500, 1000]
    # Pt_bins = [35, 50, 100, 200, 500, 2000]
    nbins = len(Pt_bins) - 1
    # optim_pars = {35: [0.49948551]}
    par = optim_pars_paths["DiTau"][1]
    deep_thr = optim_pars_paths["DiTau"][0]
    Pt_thr = Pt_thr_paths["DiTau"]

    print("Plotting differential efficiency vs gen Pt")
    num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(taus[0], taus[1],
                                                                                        gen_taus[0], gen_taus[1],
                                                                                        Pt_thr=Pt_thr)
    num_tau_mask_deepTau_1 = deepTau_selection_ptdep(taus[0], Pt_thr, par, deep_thr) & num_tau_mask_1
    num_tau_mask_deepTau_2 = deepTau_selection_ptdep(taus[1], Pt_thr, par, deep_thr) & num_tau_mask_2
    num_pair_mask = num_tau_mask_deepTau_1 & num_tau_mask_deepTau_2
    num_ev_mask = ditau_selection(num_tau_mask_deepTau_1, num_tau_mask_deepTau_2)
    # take only leading pair
    tau_num_1 = ak.firsts((taus[0][num_pair_mask])[num_ev_mask], axis=-1)
    tau_num_2 = ak.firsts((taus[1][num_pair_mask])[num_ev_mask], axis=-1)

    den_pair_mask = den_tau_mask_1 & den_tau_mask_2
    den_ev_mask = ditau_selection(den_tau_mask_1, den_tau_mask_2)
    tau_den_1 = ak.firsts((gen_taus[0][den_pair_mask])[den_ev_mask], axis=-1)
    tau_den_2 = ak.firsts((gen_taus[1][den_pair_mask])[den_ev_mask], axis=-1)

    # numerator histogram
    num_hist_2D = TH2D("num_2d", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))
    # denominator histogram
    den_hist_2D = TH2D("den_2d", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))

    # Fill histograms with gen tau pt of leading and subleading taus
    for i in range(len(tau_num_1)):
        num_hist_2D.Fill(tau_num_1.gen_pt[i], tau_num_2.gen_pt[i])
        # if i % 100 == 0:
        #     print(tau_num_1.gen_pt[i], tau_num_2.gen_pt[i])
    for i in range(len(tau_den_1)):
        den_hist_2D.Fill(tau_den_1.gen_pt[i], tau_den_2.gen_pt[i])

    # Compute efficiency
    eff_hist_2D = num_hist_2D.Clone("eff_2d")
    eff_hist_2D.Divide(den_hist_2D)

    eff_hist_2D = set_eff2Dhist_style(eff_hist_2D, Pt_thr, Pt_bins[-1])

    gStyle.SetOptStat(0)
    drawCanv_2d = TCanvas("c2", "")
    gPad.SetLogx()
    gPad.SetLogy()
    gStyle.SetPaintTextFormat("1.2f")
    eff_hist_2D.Draw("colz text")

    drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf")

    if Pt_thr == 35:
        num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(taus_base[0], taus_base[1],
                                                                                            gen_taus_base[0], gen_taus_base[1],
                                                                                            Pt_thr=Pt_thr)
        num_pair_mask = num_tau_mask_1 & num_tau_mask_2
        num_ev_mask = ditau_selection(num_tau_mask_1, num_tau_mask_2)
        # take only leading pair
        tau_num_1 = ak.firsts((taus_base[0][num_pair_mask])[num_ev_mask], axis=-1)
        tau_num_2 = ak.firsts((taus_base[1][num_pair_mask])[num_ev_mask], axis=-1)

        den_pair_mask = den_tau_mask_1 & den_tau_mask_2
        den_ev_mask = ditau_selection(den_tau_mask_1, den_tau_mask_2)
        tau_den_1 = ak.firsts((gen_taus_base[0][den_pair_mask])[den_ev_mask], axis=-1)
        tau_den_2 = ak.firsts((gen_taus_base[1][den_pair_mask])[den_ev_mask], axis=-1)

        # numerator histogram
        num_hist_2D_base = TH2D("num_2d_base", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))
        # denominator histograms
        den_hist_2D_base = TH2D("den_2d_base", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))

        # Fill histograms with gen tau pt of leading and subleading taus
        for i in range(len(tau_num_1)):
            num_hist_2D_base.Fill(tau_num_1.gen_pt[i], tau_num_2.gen_pt[i])
            # if i % 100 == 0:
            #     print(tau_num_1.gen_pt[i], tau_num_2.gen_pt[i])
        for i in range(len(tau_den_1)):
            den_hist_2D_base.Fill(tau_den_1.gen_pt[i], tau_den_2.gen_pt[i])

        # Compute efficiency
        eff_hist_2D_base = num_hist_2D_base.Clone("eff_2d_base")
        eff_hist_2D_base.Divide(den_hist_2D_base)

        eff_hist_2D_base = set_eff2Dhist_style(eff_hist_2D_base, Pt_thr, Pt_bins[-1], cut_based=True)

        drawCanv_2d = TCanvas("c2_base", "")
        gPad.SetLogx()
        gPad.SetLogy()
        gStyle.SetPaintTextFormat("1.2f")
        eff_hist_2D_base.Draw("colz text")
        drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_base_" + plot_name + ".pdf")

    hfile.Write()

    print("Total efficiency:")
    eff_deep = compute_deepTau_ptdep_eff(taus[0], taus[1], gen_taus[0], gen_taus[1], par, deep_thr, Pt_thr=Pt_thr)
    print(eff_deep)
    print("Total rate:")
    rate_deep = compute_deepTau_ptdep_rate_diTau(taus_rates[0], taus_rates[1], Nev_den, par, deep_thr, Pt_thr=Pt_thr,
                                                 L1rate=L1rate_bm)
    print(rate_deep)

