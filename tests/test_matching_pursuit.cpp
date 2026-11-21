// test_matching_pursuit.cpp  --  verify MatchingPursuit: a single atom is
// recovered exactly, energy is conserved at every step, the residual
// only shrinks, atoms + residual rebuild the signal, and the dictionary
// picks short atoms for clicks and long atoms for steady tones.

#include "MatchingPursuit.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace weft;

namespace {

int n_passed = 0, n_failed = 0;
void check(bool ok, const std::string& name) {
    if (ok) { ++n_passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++n_failed; std::cout << "  FAIL  " << name << "\n"; }
}

double energy(const std::vector<double>& x) {
    double e = 0.0;
    for (double v : x) e += v * v;
    return e;
}

const double PI = 3.14159265358979323846;

} // namespace

void test_waveform() {
    std::cout << "atom waveform:\n";
    MatchingPursuit<double> mp({64, 256});
    const auto g = mp.waveform(1, 20, 0.7);
    check(g.size() == 256, "length equals the window length");
    check(std::fabs(energy(g) - 1.0) < 1e-12, "unit energy");
    check(std::fabs(g[0]) < 1e-12, "starts at zero (Hann window)");
}

// A signal that IS one atom must come back as that atom, with nothing left.
void test_single_atom() {
    std::cout << "one atom in, one atom out:\n";
    MatchingPursuit<double> mp({64, 256, 1024});
    const std::size_t n = 4096;
    const std::size_t start = 1024;            // on the frame grid of scale 256 (hop 64)
    const auto g = mp.waveform(1, 37, 1.1);
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < g.size(); ++i) x[start + i] = 0.8 * g[i];

    std::vector<double> trace, residual;
    const auto atoms = mp.decompose(x, 1, 0.0, &trace, &residual);
    check(atoms.size() == 1, "one atom returned");
    const Atom& a = atoms[0];
    check(a.scale == 1 && a.bin == 37, "window length and frequency bin are right");
    check(a.start == int(start), "time position is right");
    check(std::fabs(a.amp - 0.8) < 1e-6, "amplitude is right");
    std::cout << "  residual energy / signal energy = " << energy(residual) / energy(x) << "\n";
    check(energy(residual) < 1e-10 * energy(x), "nothing is left in the residual");
}

// ||x||^2 = ||residual||^2 + sum a^2 at every step; the residual energy
// never increases.
void test_energy_conservation() {
    std::cout << "energy conservation on noise plus tones:\n";
    MatchingPursuit<double> mp({64, 256, 1024});
    const std::size_t n = 6000;
    std::mt19937 gen(3);
    std::uniform_real_distribution<double> d(-0.2, 0.2);
    // The signal is silent for one full window (1024 samples) at each end.
    // Atoms may overhang the ends of a signal, and the part of an atom that
    // falls outside is not in the residual that decompose() returns; with
    // silent margins no atom that touches the sound can reach outside, so
    // the identity can be checked exactly on the returned residual.
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 1024; i < n - 1024; ++i)
        x[i] = d(gen) + 0.5 * std::sin(2 * PI * 0.031 * i)
                      + 0.3 * std::sin(2 * PI * 0.117 * i + 0.4);

    std::vector<double> trace, residual;
    const auto atoms = mp.decompose(x, 200, 0.0, &trace, &residual);
    check(atoms.size() == 200 && trace.size() == 201, "200 atoms, 201 energy values");

    bool monotone = true;
    for (std::size_t i = 1; i < trace.size(); ++i)
        if (trace[i] > trace[i - 1] + 1e-12) monotone = false;
    check(monotone, "residual energy never increases");

    double sum_a2 = 0.0;
    for (const Atom& a : atoms) sum_a2 += double(a.amp) * a.amp;
    const double lhs = energy(x), rhs = energy(residual) + sum_a2;
    std::cout << "  ||x||^2 = " << lhs << ",  ||r||^2 + sum a^2 = " << rhs << "\n";
    check(std::fabs(lhs - rhs) < 1e-6 * lhs, "energy identity holds (to float precision of amp)");
    check(std::fabs(trace.back() - energy(residual)) < 1e-9 * lhs,
          "the running energy equals the residual's true energy");
    std::cout << "  energy explained by 200 atoms: "
              << 100.0 * (1.0 - trace.back() / trace[0]) << "%\n";
    check(trace.back() < 0.2 * trace[0], "200 atoms explain over 80% of the energy");
}

// synthesize(atoms) + residual == signal.
void test_reconstruction() {
    std::cout << "atoms + residual rebuild the signal:\n";
    MatchingPursuit<double> mp({64, 256, 1024});
    const std::size_t n = 5000;
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 0.6 * std::sin(2 * PI * 0.05 * i) * std::exp(-double(i) / 2000.0);
    x[2500] += 1.0;                                    // a click

    std::vector<double> residual;
    const auto atoms = mp.decompose(x, 150, 0.0, nullptr, &residual);
    const auto y = mp.synthesize(atoms, n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::fabs(y[i] + residual[i] - x[i]));
    std::cout << "  worst |synth + residual - signal| = " << worst << "\n";
    check(worst < 1e-6, "sample by sample (to float precision of amp and phase)");
}

// The reason for several window lengths: a click is best described by a
// short atom, a steady tone by a long one.
void test_scale_selection() {
    std::cout << "the dictionary adapts its window length:\n";
    MatchingPursuit<double> mp({64, 256, 1024});
    const std::size_t n = 8192;

    std::vector<double> click(n, 0.0);
    for (std::size_t i = 0; i < 12; ++i)
        click[4000 + i] = std::sin(2 * PI * 0.3 * i) * std::sin(PI * i / 12.0);
    auto a1 = mp.decompose(click, 1);
    check(!a1.empty() && a1[0].scale == 0, "a 12-sample click -> the shortest atom");
    check(!a1.empty() && std::abs(a1[0].start + 32 - 4006) < 40, "placed at the click");

    std::vector<double> tone(n);
    for (std::size_t i = 0; i < n; ++i) tone[i] = std::sin(2 * PI * 0.0625 * i);
    auto a2 = mp.decompose(tone, 1);
    check(!a2.empty() && a2[0].scale == 2, "a steady sine -> the longest atom");
    check(!a2.empty() && a2[0].bin == 64, "at the sine's frequency (bin 0.0625 * 1024)");
}

void test_stop_ratio_and_errors() {
    std::cout << "stopping rule and argument checks:\n";
    MatchingPursuit<double> mp({64, 256});
    std::vector<double> x(3000);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(2 * PI * 0.07 * i);
    std::vector<double> trace;
    const auto atoms = mp.decompose(x, 100000, 0.05, &trace);
    check(trace.back() <= 0.05 * trace[0], "stops once 95% of the energy is explained");
    check(atoms.size() < 500, "and does so with few atoms");

    bool caught = false;
    try { MatchingPursuit<double> bad({100}); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "a window length that is not a power of two is rejected");

    const auto silent = mp.decompose(std::vector<double>(2000, 0.0), 10);
    check(silent.empty(), "silence yields no atoms");
}

int main() {
    std::cout << "test_matching_pursuit\n";
    test_waveform();
    test_single_atom();
    test_energy_conservation();
    test_reconstruction();
    test_scale_selection();
    test_stop_ratio_and_errors();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
