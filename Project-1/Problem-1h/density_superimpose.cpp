// read dists_gaussian.csv and dists_repulsive.csv and build them on same canvas

#include <TFile.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TStyle.h>

#include <fstream>

int main()
{
    auto inf1 = std::ifstream("dists_gaussian.csv");
    auto inf2 = std::ifstream("dists_repulsive.csv");

    auto h1 = new TH1D("h1", "h1", 100, 0, 1.5);
    auto h2 = new TH1D("h2", "h2", 100, 0, 1.5);

    auto canvas = new TCanvas("canvas", "canvas");

    // read files line by line, and use stod to convert to double; also calculate errors with sumw2
    std::string line;
    h1->Sumw2();
    h2->Sumw2();
    while (std::getline(inf1, line))
        h1->Fill(std::stod(line));
    while (std::getline(inf2, line))
        h2->Fill(std::stod(line));

    // Normalize by counts
    h1->Scale(1.0 / h1->Integral());
    h2->Scale(1.0 / h2->Integral());

    gPad->SetTicks();
    gPad->SetTopMargin(0.05);
    gPad->SetLeftMargin(0.11);
    gPad->SetRightMargin(0.05);
    gPad->SetBottomMargin(0.1);
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    // Superimpose
    h1->SetLineColor(kRed);
    h1->Draw("HIST E");
    h1->GetXaxis()->SetTitle("r/a_{0}");
    h1->GetYaxis()->SetTitle("Density normalized");

    h2->SetLineColor(kBlue);
    h2->Draw("HIST E SAME");

    // Save to file
    canvas->SaveAs("density_superimpose.pdf");
}