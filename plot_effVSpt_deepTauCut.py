import json
import ROOT
from ROOT import *
from array import array
from scipy import optimize
import argparse
from common.selection import *
from common.eff_rate import *
from common.dataset import dataset


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
    dataset_eff = dataset(data_path + fileName_eff, treeName_in, treeName_gen)
    taus = dataset_eff.get_taus()
    gen_taus = dataset_eff.get_gen_taus()

    # get sample for rates
    print("Loading sample for rate")
    dataset_rates = dataset(data_path + fileName_rates, treeName_in, treeName_gen)
    taus_rates = dataset_rates.get_taus()
    Nev_den = len(dataset_rates.get_gen_events())

    Pt_thr_list = [20, 25, 30, 35, 40, 45]
    Pt_bins = [20, 25, 30, 35, 40, 45, 50, 60, 70, 100, 200]
    nbins = len(Pt_bins) - 1
    thr_list = np.flip(np.linspace(0.0, 1.0, num=101))
    eff_atThreshold = []

    for n, Pt_thr in enumerate(Pt_thr_list):

        def f(deepTau_thr):
            rate, _, _ = compute_deepTau_rate(taus_rates, Nev_den, deepTau_thr, Pt_thr=Pt_thr, L1rate=L1rate_bm)
            isocut_rate, _, _ = compute_isocut_rate(taus_rates, Nev_den, "mediumIsoAbs", "mediumIsoRel", Pt_thr=35)
            return rate - isocut_rate

        solution = optimize.root_scalar(f, bracket=[0, 1], method='bisect')
        if solution.converged:
            deepTau_thr = solution.root
            print("deepTau threshold at {} GeV: {}".format(Pt_thr, deepTau_thr))
        else:
            print("root finding did not converged for Pt {} GeV".format(Pt_thr))

        print("Plotting differential efficiency vs gen Pt")
        num_tau_mask, den_tau_mask = num_den_mask_eff(taus, gen_taus, Pt_thr=Pt_thr)
        num_tau_mask_deepTau = deepTau_selection(taus, deepTau_thr) & num_tau_mask
        num_ev_mask = ditau_selection(num_tau_mask_deepTau)
        taus_num = taus[num_tau_mask_deepTau]
        taus_num = taus_num[num_ev_mask]

        den_ev_mask = ditau_selection(den_tau_mask)
        taus_den = gen_taus[den_tau_mask]
        taus_den = taus_den[den_ev_mask]

        # numerator histograms
        num_hist = TH1D("num", "", nbins, array("d", Pt_bins))
        num_hist_2D = TH2D("num_2d", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))
        # denominator histograms
        den_hist = TH1D("den", "", nbins, array("d", Pt_bins))
        den_hist_2D = TH2D("den_2d", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))

        # Fill histograms with gen tau pt of leading and subleading taus
        for gen_pt in taus_num.gen_pt:
            gen_pt_sorted = np.sort(gen_pt)[::-1]
            num_hist.Fill(gen_pt_sorted[0])
            num_hist.Fill(gen_pt_sorted[1])
            num_hist_2D.Fill(gen_pt_sorted[0], gen_pt_sorted[1])
        for gen_pt in taus_den.gen_pt:
            gen_pt_sorted = np.sort(gen_pt)[::-1]
            den_hist.Fill(gen_pt_sorted[0])
            den_hist.Fill(gen_pt_sorted[1])
            den_hist_2D.Fill(gen_pt_sorted[0], gen_pt_sorted[1])

        # Compute efficiency
        eff_hist = num_hist.Clone("eff")
        eff_hist_2D = num_hist_2D.Clone("eff_2d")
        eff_hist.Divide(den_hist)
        eff_hist_2D.Divide(den_hist_2D)

        eff_hist.SetTitle("Pt > {}".format(Pt_thr))
        eff_hist.GetXaxis().SetTitle("gen p_{T} [GeV]")
        eff_hist.GetXaxis().SetRangeUser(Pt_bins[0], Pt_bins[-1]-5.)
        eff_hist.GetYaxis().SetTitle("Efficiency")

        eff_hist_2D = set_eff2Dhist_style(eff_hist_2D, Pt_thr, Pt_bins[-1])

        if Pt_thr==35:
            num_tau_mask_mediumIso = iso_tau_selection(taus, "mediumIsoAbs", "mediumIsoRel") & num_tau_mask
            num_ev_mask = ditau_selection(num_tau_mask_mediumIso)
            taus_num = taus[num_tau_mask_mediumIso]
            taus_num = taus_num[num_ev_mask]
            num_hist_2D = TH2D("num_2d", "", nbins, array("d", Pt_bins), nbins, array("d", Pt_bins))
            for gen_pt in taus_num.gen_pt:
                gen_pt_sorted = np.sort(gen_pt)[::-1]
                num_hist_2D.Fill(gen_pt_sorted[0], gen_pt_sorted[1])
            eff_hist_2D_base = num_hist_2D.Clone("eff_2d_base")
            eff_hist_2D_base.Divide(den_hist_2D)
            eff_hist_2D_base = set_eff2Dhist_style(eff_hist_2D_base, Pt_thr, Pt_bins[-1], cut_based=True)

        gStyle.SetOptStat(0)
        drawCanv = TCanvas("c","")
        gPad.SetLogx(0)
        gPad.SetLogy(0)
        eff_hist.Draw("E")
        drawCanv_2d = TCanvas("c2", "")
        gPad.SetLogx()
        gPad.SetLogy()
        gStyle.SetPaintTextFormat("1.2f")
        eff_hist_2D.Draw("colz text")

        if n == 0 and n != len(Pt_thr_list) - 1:
            drawCanv.Print(plot_path + "diffeff_VSgenPt_" + plot_name + ".pdf(")
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf(")
        elif n == len(Pt_thr_list) - 1:
            drawCanv.Print(plot_path + "diffeff_VSgenPt_" + plot_name + ".pdf)")
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf)")
        else:
            drawCanv.Print(plot_path + "diffeff_VSgenPt_" + plot_name + ".pdf")
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_" + plot_name + ".pdf")

        if Pt_thr==35:
            drawCanv_2d = TCanvas("c2_base", "")
            gPad.SetLogx()
            gPad.SetLogy()
            gStyle.SetPaintTextFormat("1.2f")
            eff_hist_2D_base.Draw("colz text")
            drawCanv_2d.Print(plot_path + "diffeff_VSgenPt2D_base_" + plot_name + ".pdf")


    # # plt.title(r"Efficiency vs $p_{T}$")
    # plt.xlabel(r"$p_{T}$ threshold [GeV]")
    # plt.ylabel("Efficiency")
    # plt.plot(Pt_thr_list, eff_atThreshold, ".", label="deepTau discriminator")
    # eff_base, _, _ = compute_isocut_eff(taus, gen_taus, "mediumIsoAbs", "mediumIsoRel", Pt_thr=35.)
    # print("\nBase efficiency", eff_base)
    # plt.plot(35, eff_base, ".", color="orange", label="cut-based Medium WP (Run 2 setup)")
    # plt.legend()
    # plt.savefig(plot_path + "effVSpt_" + plot_name + ".pdf")
    # plt.close()
