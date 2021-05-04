import json
import matplotlib.pyplot as plt
import ROOT
from ROOT import *
from array import array
from scipy import optimize
import argparse
from common.selection import *
from common.eff_rate import *
from common.dataset import Dataset


def set_eff2Dhist_style(hist, Pt_thr, Pt_max, cut_based = False):
    if cut_based:
        hist.SetTitle("Cut-based Medium Efficiency (Run 2 setup, p_{T} > %d GeV)" % Pt_thr)
    else:
        hist.SetTitle("Efficiency for p_{T} > %d GeV" % Pt_thr)
    hist.GetXaxis().SetTitle("gen p_{T} leading tau [GeV]")
    hist.GetYaxis().SetTitle("gen p_{T} subleading tau [GeV]")
    hist.GetXaxis().SetMoreLogLabels(kTRUE)
    hist.GetYaxis().SetMoreLogLabels(kTRUE)
    hist.GetYaxis().SetTitleOffset(1.2)
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
    plot_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/plots/newPlots_CMSSW_11_2_0/"
    data_path = "/Users/mascella/workspace/EPR-workspace/analysis_deepTau/data/newSamples_CMSSW_11_2_0/"
    fileName_eff = "VBFHToTauTau.root"
    fileName_rates = "EphemeralHLTPhysics_1to8.root"
    treeName_gen = "gen_counter"
    treeName_in = "final_counter"

    # L1 rate
    L1rate = 75817.94
    lumi_bm = 2e-2
    lumi_real = 122.792/7319
    L1rate_bm = L1rate * lumi_bm/lumi_real

    # get sample for efficiency
    print("Loading sample for efficiency")
    dataset_eff = Dataset(data_path + fileName_eff, treeName_in, treeName_gen)
    taus = dataset_eff.get_tau_pairs()
    gen_taus = dataset_eff.get_gen_tau_pairs()

    # get sample for rates
    print("Loading sample for rate")
    dataset_rates = Dataset(data_path + fileName_rates, treeName_in, treeName_gen)
    taus_rates = dataset_rates.get_tau_pairs()
    Nev_den = len(dataset_rates.get_gen_events())

    Pt_thr_list = [20, 25, 30, 35, 40, 45]
    Pt_bins = [20, 25, 30, 35, 40, 45, 50, 60, 70, 100, 200]
    nbins = len(Pt_bins) - 1
    eff_atThreshold = np.zeros((3, len(Pt_thr_list)))

    for n, Pt_thr in enumerate(Pt_thr_list):

        def f(deepTau_thr):
            rate, _, _ = compute_deepTau_rate(taus_rates[0], taus_rates[1], Nev_den, deepTau_thr, Pt_thr=Pt_thr, L1rate=L1rate_bm)
            isocut_rate, _, _ = compute_isocut_rate(taus_rates[0], taus_rates[1], Nev_den, "mediumIsoAbs", "mediumIsoRel", Pt_thr=35)
            return rate - isocut_rate

        solution = optimize.root_scalar(f, bracket=[0, 1], method='bisect')
        if solution.converged:
            thr = solution.root
            print("deepTau threshold at {} GeV: {}".format(Pt_thr, thr))
        else:
            print("root finding did not converged for Pt {} GeV".format(Pt_thr))
            break
        eff_atThreshold[:, n] = compute_deepTau_eff(taus[0], taus[1], gen_taus[0], gen_taus[1], thr, Pt_thr=Pt_thr)

        print("Plotting differential efficiency vs gen Pt")
        num_tau_mask_1, num_tau_mask_2, den_tau_mask_1, den_tau_mask_2 = apply_numden_masks(taus[0], taus[1], gen_taus[0], gen_taus[1], Pt_thr=Pt_thr)
        num_tau_mask_deepTau_1 = deepTau_selection(taus[0], thr) & num_tau_mask_1
        num_tau_mask_deepTau_2 = deepTau_selection(taus[1], thr) & num_tau_mask_2
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
        # denominator histograms
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

        if Pt_thr==35:
            num_tau_mask_mediumIso_1 = iso_tau_selection(taus[0], "mediumIsoAbs", "mediumIsoRel") & num_tau_mask_1
            num_tau_mask_mediumIso_2 = iso_tau_selection(taus[1], "mediumIsoAbs", "mediumIsoRel") & num_tau_mask_2
            num_pair_mask = num_tau_mask_mediumIso_1 & num_tau_mask_mediumIso_2
            num_ev_mask = ditau_selection(num_tau_mask_mediumIso_1, num_tau_mask_mediumIso_2)

            tau_num_1 = ak.firsts((taus[0][num_pair_mask])[num_ev_mask], axis=-1)
            tau_num_2 = ak.firsts((taus[1][num_pair_mask])[num_ev_mask], axis=-1)
            num_hist_2D = TH2D("num_2d", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))
            for i in range(len(tau_num_1)):
                num_hist_2D.Fill(tau_num_1.gen_pt[i], tau_num_2.gen_pt[i])
            eff_hist_2D_base = num_hist_2D.Clone("eff_2d_base")
            eff_hist_2D_base.Divide(den_hist_2D)
            eff_hist_2D_base = set_eff2Dhist_style(eff_hist_2D_base, Pt_thr, Pt_bins[-1], cut_based=True)

        gStyle.SetOptStat(0)
        drawCanv_2d = TCanvas("c2", "")
        gPad.SetLogx()
        gPad.SetLogy()
        gStyle.SetPaintTextFormat("1.2f")
        eff_hist_2D.Draw("colz text")

        if n == 0 and n != len(Pt_thr_list) - 1:
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf(")
        elif n == len(Pt_thr_list) - 1:
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf)")
        else:
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf")

        if Pt_thr==35:
            drawCanv_2d = TCanvas("c2_base", "")
            gPad.SetLogx()
            gPad.SetLogy()
            gStyle.SetPaintTextFormat("1.2f")
            eff_hist_2D_base.Draw("colz text")
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_base_" + plot_name + ".pdf")


    # plt.title(r"Efficiency vs $p_{T}$")
    plt.xlabel(r"$p_{T}$ threshold [GeV]")
    plt.ylabel("Efficiency")
    plt.errorbar(Pt_thr_list, eff_atThreshold[0, :], yerr= eff_atThreshold[1:, :], marker='.', label="deepTau discriminator", linestyle="")
    eff_base, _, _ = compute_isocut_eff(taus[0], taus[1], gen_taus[0], gen_taus[1], "mediumIsoAbs", "mediumIsoRel", Pt_thr=35.)
    print("\nBase efficiency", eff_base)
    print(eff_atThreshold[0, :])
    plt.plot(35, eff_base, ".", color="orange", label="cut-based Medium WP (Run 2 setup)")
    plt.legend()
    plt.savefig(plot_path + "effVSpt_" + plot_name + ".pdf")
    plt.close()
