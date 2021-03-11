import sys
from helpers import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common.dataset import dataset
import json

plot_name = sys.argv[1]
plot_path = '/plots/minPt20_pre10/'
data_path = '/data/'
QCD_fileJson = "QCD_samples.json"

isocut_vars = {
                 "loose": ["looseIsoAbs", "looseIsoRel"],
                 "medium": ["mediumIsoAbs", "mediumIsoRel"],
                 "tight": ["tightIsoAbs", "tightIsoRel"]
                }
colors = ["green", "red", "orange"]

# get VBF sample
treeName_gen = "gen_counter"
treeName_in = "initial_counter"

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
print(QCD_den_list)
print(QCD_xs_list)
lumi = 128.91


with PdfPages(plot_path + 'QCDplots_{}.pdf'.format(plot_name)) as pdf:

    pts_list = np.array([])
    weights_list = np.array([])
    dts_list = np.array([])
    for i, QCD_taus in enumerate(QCD_taus_list):
        dts = QCD_taus.deepTau_VSjet.flatten()
        pts = QCD_taus.pt.flatten()
        scale = QCD_xs_list[i]*lumi/QCD_den_list[i]
        weights = np.ones(len(pts))*scale
        dts_list = np.append(dts_list, dts)
        pts_list = np.append(pts_list, pts)
        weights_list = np.append(weights_list, weights)
    plt.xlabel(r"$p_{T}$ [GeV]")  
    plt.hist(pts_list, weights=weights_list, bins=100, range=[0,100])
    pdf.savefig()
    plt.close()
    plt.xlabel("deepTau_VSjet")
    plt.hist(dts_list, weights=weights_list, bins=100)
    pdf.savefig()
    plt.close()
        