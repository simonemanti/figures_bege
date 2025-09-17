import matplotlib.pyplot as plt
import ROOT
import numpy as np
import os, sys, pickle
from scipy.special import erf

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)
from rcparams import plotter, textwidth, columnwidth

@plotter()
def main_fit():

    ROOT.gStyle.SetOptStat(0)

    with open('../Calibrated_data/data21_with_label_v2.pkl', 'rb') as f:
        df = pickle.load(f)
    
    def histo(nbins, xmin, xmax, h_name, title="",  xlabel="", ylabel=""):
        existing_hist = ROOT.gROOT.FindObject(h_name)
        if existing_hist:
            existing_hist.Delete()
        h = ROOT.TH1F(h_name, title, nbins, xmin, xmax)
        h.GetXaxis().SetTitle(xlabel)
        h.GetYaxis().SetTitle(ylabel)
        return h

    def line(x, y, col, style=1):
        l = ROOT.TLine(x, 0, x, y)
        l.SetLineWidth(2)
        l.SetLineColor(col)
        l.SetLineStyle(style)
        l.Draw('same')
        return l
    
    def gauss_integral_and_error(A, mu, sigma, x1, x2, err_A, err_mu, err_sigma, cov_A_mu=0, cov_A_sigma=0, cov_mu_sigma=0):
        sqrt2 = np.sqrt(2)
        sqrt2pi = np.sqrt(2 * np.pi)
        u1 = (x1 - mu) / (sqrt2 * sigma)
        u2 = (x2 - mu) / (sqrt2 * sigma)
        F = 0.5 * (erf(u2) - erf(u1))
        I = A * sigma * sqrt2pi * F

        # Partial derivatives
        exp1 = np.exp(-0.5 * ((x1 - mu) / sigma)**2)
        exp2 = np.exp(-0.5 * ((x2 - mu) / sigma)**2)
        dF_dmu = -(exp2 - exp1) / (sqrt2pi * sigma)
        dF_dsigma = ((x2 - mu) * exp2 - (x1 - mu) * exp1) / (sqrt2pi * sigma**2)
        
        dI_dA = sigma * sqrt2pi * F
        dI_dmu = A * sigma * sqrt2pi * dF_dmu
        dI_dsigma = A * sqrt2pi * (F + sigma * dF_dsigma)

        # Error propagation
        var_I = (
            (dI_dA * err_A)**2 +
            (dI_dmu * err_mu)**2 +
            (dI_dsigma * err_sigma)**2 +
            2 * dI_dA * dI_dmu * cov_A_mu +
            2 * dI_dA * dI_dsigma * cov_A_sigma +
            2 * dI_dmu * dI_dsigma * cov_mu_sigma
        )
        err_I = np.sqrt(var_I)
        return I, err_I

    # Numerical derivatives
    def numerical_derivative(bg_func, params, i, delta, x_min, x_max):
        params1 = params[:]
        params1[i] += delta
        for j, v in enumerate(params1):
            bg_func.SetParameter(j, v)
        plus = bg_func.Integral(x_min, x_max)
        params1[i] -= 2*delta
        for j, v in enumerate(params1):
            bg_func.SetParameter(j, v)
        minus = bg_func.Integral(x_min, x_max)
        for j, v in enumerate(params):  # reset
            bg_func.SetParameter(j, v)
        return (plus - minus) / (2*delta)
    

    mask = (df['pred_label']==1)
    spectrum2 = df[mask]['energy'].values

    nbins = 1000
    hist_area2 = histo(nbins, 0, 1000, "hist_with_ML", "", "keV", f"Counts / 1 keV")

    for idx, i in enumerate(spectrum2):
        hist_area2.Fill(i)
    
    c = ROOT.TCanvas()
    c.SetCanvasSize(1200,800)
    c.Draw()

    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.2, 1, 1.0)
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.2)
    pad1.SetTopMargin(0.012)
    pad1.SetBottomMargin(0.012)
    pad1.SetLeftMargin(0.08)
    pad1.SetRightMargin(0.01)

    pad2.SetTopMargin(0.01)
    pad2.SetBottomMargin(0.45)
    pad2.SetLeftMargin(0.08)
    pad2.SetRightMargin(0.01)
    pad1.Draw()
    pad2.Draw()

    pad1.cd()

    lower = 15
    upper = 650


    bkg = "expo(0) * 0.5 * (1 + TMath::Erf((x - [2])/[3])) + expo(25) * 0.5 * (1 + TMath::Erf((x - [27])/[28]))"
    fit = ROOT.TF1("fit",
        f'{bkg}'
        " + gaus(4)+gaus(7)+gaus(10)+gaus(13)+gaus(16)+gaus(19)+gaus(22)",
        lower, upper)

    # ---- background ----
    fit.SetParameter(2, 100)    # erf center (estimate from your spectrum)
    fit.SetParameter(3, 30)     # erf width (estimate)
    fit.SetParLimits(2, 80, 130)
    fit.SetParLimits(3, 10, 200)

    fit.SetParameter(25, 1)    # amplitude for expo(25)
    fit.SetParameter(26, 0.01) # slope for expo(25)
    fit.SetParLimits(25, 0, 1)  # or suitable limits
    fit.SetParLimits(26, -1, 0)

    fit.SetParameter(27, 300)  # erf center (guess)
    fit.SetParameter(28, 30)   # erf width (guess)
    fit.SetParLimits(27, 200, 350)
    fit.SetParLimits(28, 10, 200)

    # ---- right peaks ----
    fit.SetParameter(4, 40)     # amplitude for mean=25
    fit.SetParameter(5, 25)     # mean
    fit.SetParameter(6, 1.5)    # sigma
    fit.SetParLimits(4, 35, 50)
    fit.SetParLimits(5, 23, 28)
    fit.SetParLimits(6, 0.1, 5)

    fit.SetParameter(7, 150)    # amplitude for mean=43
    fit.SetParameter(8, 4)
    fit.SetParameter(9, 0.3)
    fit.SetParLimits(7, 130, 200)
    fit.SetParLimits(8, 44.4, 45)
    fit.SetParLimits(9, 0, 3)

    fit.SetParameter(10, 37)    # amplitude for mean=71
    fit.SetParameter(11, 71)
    fit.SetParameter(12, 0.3)
    fit.SetParLimits(10, 15, 50)
    fit.SetParLimits(11, 72, 76)
    fit.SetParLimits(12, 0, 3)

    fit.SetParameter(13, 50)    # amplitude for mean=237
    fit.SetParameter(14, 237)
    fit.SetParameter(15, 2.3)
    fit.SetParLimits(13, 25, 70)
    fit.SetParLimits(14, 234, 240)
    fit.SetParLimits(15, 0, 3)

    fit.SetParameter(16, 40)    # amplitude for mean=291
    fit.SetParameter(17, 292)
    fit.SetParameter(18, 0.3)
    fit.SetParLimits(16, 20, 100)
    fit.SetParLimits(17, 291, 294.5)
    fit.SetParLimits(18, 0, 3)

    fit.SetParameter(19, 42)    # amplitude for mean=347
    fit.SetParameter(20, 348.5)
    fit.SetParameter(21, 0.3)
    fit.SetParLimits(19, 30, 70)
    fit.SetParLimits(20, 347, 349.5)
    fit.SetParLimits(21, 0, 1)

    fit.SetParameter(22, 27)    # amplitude for mean=597
    fit.SetParameter(23, 597)
    fit.SetParameter(24, 0.3)
    fit.SetParLimits(22, 25, 50)
    fit.SetParLimits(23, 603, 607.5)
    fit.SetParLimits(24, 0, 1)

    fit.SetNpx(835)
    fitResult = hist_area2.Fit(fit,"LRS")
    cov = fitResult.GetCovarianceMatrix()
    hist_area2.GetXaxis().SetLabelSize(0)
    hist_area2.GetYaxis().SetTitleSize(0.07)
    hist_area2.GetYaxis().SetTitleOffset(0.5)
    hist_area2.GetXaxis().SetRangeUser(0,650)
    hist_area2.GetYaxis().SetRangeUser(0,190)
    hist_area2.GetYaxis().CenterTitle(True)
    hist_area2.GetXaxis().SetRangeUser(0,upper)
    hist_area2.GetYaxis().SetRangeUser(0,190)
    hist_area2.Draw()

    chi2 = fit.GetChisquare()
    ndf = fit.GetNDF()
    chi2_ndf = chi2 / ndf if ndf != 0 else None

    bg_func = ROOT.TF1("bg_func",bkg, lower, upper)
    bg_func = ROOT.TF1("bg_func","expo(0) * 0.5 * (1 + TMath::Erf((x - [2])/[3])) + expo(4) * 0.5 * (1 + TMath::Erf((x - [6])/[7]))", lower, upper)
    for i, j in enumerate([0,1,2,3,25,26,27,28]):
        bg_func.SetParameter(i, fit.GetParameter(j))
        
    g1_func = ROOT.TF1("g1_func", "gaus", lower, upper)
    for i, j in enumerate([4, 5, 6]):
        g1_func.SetParameter(i, fit.GetParameter(j))

    g2_func = ROOT.TF1("g2_func", "gaus", lower, upper)
    for i, j in enumerate([7, 8, 9]):
        g2_func.SetParameter(i, fit.GetParameter(j))

    g3_func = ROOT.TF1("g3_func", "gaus", lower, upper)
    for i, j in enumerate([10, 11, 12]):
        g3_func.SetParameter(i, fit.GetParameter(j))

    g4_func = ROOT.TF1("g4_func", "gaus", lower, upper)
    for i, j in enumerate([13, 14, 15]):
        g4_func.SetParameter(i, fit.GetParameter(j))

    g5_func = ROOT.TF1("g5_func", "gaus", lower, upper)
    for i, j in enumerate([16, 17, 18]):
        g5_func.SetParameter(i, fit.GetParameter(j))

    g6_func = ROOT.TF1("g6_func", "gaus", lower, upper)
    for i, j in enumerate([19, 20, 21]):
        g6_func.SetParameter(i, fit.GetParameter(j))

    g7_func = ROOT.TF1("g7_func", "gaus", lower, upper)
    for i, j in enumerate([22, 23, 24]):
        g7_func.SetParameter(i, fit.GetParameter(j))


    bg_func.SetLineColor(ROOT.kGreen+2) 
    bg_func.SetLineStyle(2)
    bg_func.SetLineWidth(3)
    bg_func.Draw("same")

    g1_func.SetLineColor(ROOT.kBlue)
    g1_func.SetLineStyle(2)
    g1_func.SetLineWidth(2)
    g1_func.Draw("same")

    g2_func.SetLineColor(ROOT.kMagenta)
    g2_func.SetLineStyle(2)
    g2_func.SetLineWidth(2)
    g2_func.Draw("same")

    g3_func.SetLineColor(ROOT.kCyan+2)
    g3_func.SetLineStyle(2)
    g3_func.SetLineWidth(2)
    g3_func.Draw("same")

    g4_func.SetLineColor(ROOT.kOrange)
    g4_func.SetLineStyle(2)
    g4_func.SetLineWidth(2)
    g4_func.Draw("same")

    g5_func.SetLineColor(ROOT.kRed)
    g5_func.SetLineStyle(2)
    g5_func.SetLineWidth(2)
    g5_func.Draw("same")

    g6_func.SetLineColor(ROOT.kBlack)
    g6_func.SetLineStyle(2)
    g6_func.SetLineWidth(2)
    g6_func.Draw("same")

    g7_func.SetLineColor(ROOT.kViolet)
    g7_func.SetLineStyle(2)
    g7_func.SetLineWidth(2)
    g7_func.Draw("same")

    g1_func.SetRange(0,40)
    g2_func.SetRange(40,60)
    g3_func.SetRange(60,80)
    g4_func.SetRange(220,240)
    g5_func.SetRange(280,300)
    g6_func.SetRange(340,360)
    g7_func.SetRange(600,610)


    # Create a legend for the isotopes
    legend = ROOT.TLegend(0.7, 0.4, 0.9, 0.6)  # x1, y1, x2, y2
    legend.SetBorderSize(0)
    legend.SetFillStyle(0) 
    legend.SetTextSize(0.07)

    legend.AddEntry(hist_area2, 'Data')
    legend.AddEntry(fit, "Fit", 'l')
    legend.Draw('same')

    c.Update()
    # ---------------------
    pad2.cd()

    pull_hist = hist_area2.Clone("pull_hist")
    pull_hist.Reset()
    pull_hist.SetTitle("")

    for i in range(1, hist_area2.GetNbinsX()+1):
        x = hist_area2.GetBinCenter(i)
        data = hist_area2.GetBinContent(i)
        err = hist_area2.GetBinError(i)
        fitval = fit.Eval(x)
        pull = 0
        if err > 0:
            pull = (data - fitval) / err
        pull_hist.SetBinContent(i, pull)
        pull_hist.SetBinError(i, err / fit.Eval(x))

    pull_hist.SetStats(0)
    pull_hist.SetMarkerStyle(20)
    pull_hist.SetMarkerSize(0.7)
    pull_hist.SetLineColor(ROOT.kBlue)
    pull_hist.GetYaxis().SetTitle("Pull")
    pull_hist.GetYaxis().SetTitleSize(0.3)
    pull_hist.GetYaxis().SetTitleOffset(0.1)
    pull_hist.GetYaxis().SetLabelSize(0.11)
    pull_hist.GetYaxis().SetRangeUser(-3,3)

    pull_hist.GetXaxis().SetTitleSize(0.3)
    pull_hist.GetXaxis().SetTitleOffset(0.65)
    pull_hist.GetXaxis().SetLabelSize(0.12)
    pull_hist.GetXaxis().SetLabelOffset(0.05)
    pull_hist.GetXaxis().SetTitle("keV")
    pull_hist.Draw("PE")
    pull_hist.SetLineColor(ROOT.kBlack)
    pull_hist.GetXaxis().CenterTitle(False)
    pull_hist.GetYaxis().CenterTitle(True)

    # line_plus3 = ROOT.TLine(lower, 3, 650, 3)
    # line_plus3.SetLineStyle(2)  # dashed
    # line_plus3.SetLineWidth(2)
    # line_plus3.Draw("same")

    # line_minus3 = ROOT.TLine(lower, -3, 650, -3)
    # line_minus3.SetLineStyle(2)  # dashed
    # line_minus3.SetLineWidth(2)
    # line_minus3.Draw("same")
    pull_hist.GetXaxis().SetRangeUser(0,650)

    # Optional: Draw zero line
    zero = ROOT.TLine(hist_area2.GetXaxis().GetXmin(), 0, 650, 0)
    zero.SetLineColor(ROOT.kRed)
    zero.SetLineWidth(3)
    ROOT.gPad.SetGridy() 
    zero.Draw("same")

    c.cd()
    c.Update()

    # ------- calculate s/sqrt b --------
    g_funcs = [bg_func, g1_func, g2_func, g3_func, g4_func, g5_func, g6_func, g7_func]

    peak_names = [
        "210-Pb",
        "__",
        "212-Pb",
        "214-Pb",
        "214-Pb",
        "214-Bi"
    ]

    n = 2  # or whatever your n is for integration range

    s_mean=[]
    for j, (func, name) in enumerate(zip(g_funcs[2:], peak_names)):
        temp_func = ROOT.TF1("gtemp_func", "gaus", lower, upper)
        temp_func.SetParameter(0, func.GetParameter(0))
        temp_func.SetParameter(1, func.GetParameter(1))
        temp_func.SetParameter(2, func.GetParameter(2))
        mu = temp_func.GetParameter(1)
        sigma = temp_func.GetParameter(2)
        x_min = mu - n*sigma
        x_max = mu + n*sigma

        s = temp_func.Integral(x_min, x_max)
        b = bg_func.Integral(x_min, x_max)
        significance = s / np.sqrt(b) if b > 0 else 0

        print(f"{name}: S = {s:.2f}, B = {b:.1f}, S/sqrt(B) = {significance:.2f}")
        s_mean.append(significance)

        if name == '210-Pb':
            fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
            resolution = (fwhm / mu) * 100
            print(f"resolution (FWHM) using 210-Pb = {fwhm:.2f} keV")
            print(f"resolution using 210-Pb = {resolution:.2f}%")


    # ================================= Signal =================================
    j = 7  # index for your Gaussian
    A = fit.GetParameter(j)
    mu = fit.GetParameter(j+1)
    sigma = fit.GetParameter(j+2)
    err_A = fit.GetParError(j)
    err_mu = fit.GetParError(j+1)
    err_sigma = fit.GetParError(j+2)
    cov = fitResult.GetCovarianceMatrix()
    cov_A_mu = cov[j][j+1]
    cov_A_sigma = cov[j][j+2]
    cov_mu_sigma = cov[j+1][j+2]

    x1 = mu - 2*sigma
    x2 = mu + 2*sigma

    S, sigma_S = gauss_integral_and_error(A, mu, sigma, x1, x2, err_A, err_mu, err_sigma, cov_A_mu, cov_A_sigma, cov_mu_sigma)
    print(f"S 210-Pb = {S:.2f} ± {sigma_S:.2f}")

    # ================================= Background =================================
    fit_indices = [0, 1, 2, 3, 25, 26, 27, 28]
    params_bg = [fit.GetParameter(i) for i in fit_indices]
    B = bg_func.Integral(x1, x2)

    delta = 1e-4
    derivs = []
    for i in range(len(params_bg)):
        dBdpi = numerical_derivative(bg_func, params_bg, i, delta, x1, x2)
        derivs.append(dBdpi)

    n_bg = len(fit_indices)
    cov_bg = np.zeros((n_bg, n_bg))
    for ii, gi in enumerate(fit_indices):
        for jj, gj in enumerate(fit_indices):
            cov_bg[ii, jj] = cov[gi, gj]

    var_B = 0.0
    for i in range(n_bg):
        for j in range(n_bg):
            var_B += derivs[i] * derivs[j] * cov_bg[i, j]

    sigma_B = np.sqrt(var_B)

    # ================================= Significance and error =================================
    if B > 0:
        Z = S / np.sqrt(B)
        sigma_Z = np.sqrt( (sigma_S/np.sqrt(B))**2 + (S**2/(4*B**3)) * sigma_B**2 )
    else:
        Z = 0
        sigma_Z = 0

    print(f"S/sqrt(B) 210-Pb = {Z:.2f} ± {sigma_Z:.2f}")


    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.54, 0.74, f"#bf{{^{{210}}Pb FWHM = {fwhm:.2f} keV}}")
    latex.DrawLatex(0.54, 0.82, f"#bf{{^{{210}}Pb S/#sqrt{{B}} = {s_mean[0]:.2f} +/- {sigma_Z:.2f}}}")
    latex.DrawLatex(0.54, 0.9, "#it{VIP 2021}")

    latex.DrawLatex(0.17, 0.92, "#font[42]{#it{^{210}Pb}}")
    latex.DrawLatex(0.35, 0.62, "#font[42]{#it{^{212}Pb}}")
    latex.DrawLatex(0.44, 0.5, "#font[42]{#it{^{214}Pb}}")
    latex.DrawLatex(0.51, 0.56, "#font[42]{#it{^{214}Pb}}")
    latex.DrawLatex(0.86, 0.38, "#font[42]{#it{^{214}Bi}}")

    c.SaveAs(f'{os.path.basename(os.getcwd())}_fit.png')
    c.SaveAs(f'{os.path.basename(os.getcwd())}_fit.pdf')
    


def main_total_spec():

    with open('../Calibrated_data/data21_with_label_v2.pkl', 'rb') as f:
        df = pickle.load(f)
    
    mask = (df['pred_label']==1)
    spectrum = df['energy'].values
    spectrum2 = df[mask]['energy'].values
    spectrum2_bad = df[~mask]['energy'].values

    fig, ax = plt.subplots(figsize=(textwidth,textwidth*0.6), nrows=1, ncols=1, dpi=130)

    ax.plot()
    xbins = np.linspace(0,1000,1000)
    ax.hist(spectrum, bins=xbins, histtype='step', label='Total')
    ax.hist(spectrum2, bins=xbins, histtype='step', label='Good')
    ax.hist(spectrum2_bad, bins=xbins, histtype='step', label='Bad')

    ax.set_xlim(0,650)
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Count/1 keV')

    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{os.path.basename(os.getcwd())}.png',  bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf')
    plt.show()

if __name__ == '__main__':
    main_fit()
    main_total_spec()
