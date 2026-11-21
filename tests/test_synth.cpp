// test_synth.cpp  --  verify the differentiable synthesiser and its loss:
// HarmonicSynth, FilteredNoise, SpectralLoss and the pitch helpers.
// Every backward pass is checked against finite differences in double
// precision.

#include "FilteredNoise.h"
#include "HarmonicSynth.h"
#include "Matrix.h"
#include "Pitch.h"
#include "SpectralLoss.h"

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

std::vector<double> rand_vec(std::size_t n, unsigned seed, double lo = -1, double hi = 1) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<double> v(n);
    for (double& x : v) x = d(gen);
    return v;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

const double PI = 3.14159265358979323846;

} // namespace

// ---- HarmonicSynth -------------------------------------------------------

// One harmonic, constant amplitude and pitch: the output is a plain sine.
void test_harmonic_pure_sine() {
    std::cout << "HarmonicSynth, one harmonic is a sine:\n";
    const double sr = 8000.0, f = 250.0, amp = 0.7;
    const std::size_t hop = 64, n = 1000;
    const std::size_t Tn = HarmonicSynth<double>::frames_for(n, hop);
    HarmonicSynth<double> synth(sr, hop);
    std::vector<double> f0(Tn, f);
    Matrix<double> A(1, Tn, amp);
    auto y = synth.forward(f0, A, n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        worst = std::max(worst, std::fabs(y[i] - amp * std::sin(2 * PI * f * i / sr)));
    check(y.size() == n, "output has n_samples samples");
    check(worst < 1e-9, "matches amp * sin(2 pi f n / sr)");
}

// Harmonics at or above the Nyquist frequency must be silent.
void test_harmonic_nyquist() {
    std::cout << "HarmonicSynth, no aliasing:\n";
    const double sr = 8000.0, f = 1500.0;       // harmonics 1.5k, 3k, [4.5k, 6k]
    const std::size_t hop = 32, n = 512;
    const std::size_t Tn = HarmonicSynth<double>::frames_for(n, hop);
    HarmonicSynth<double> synth(sr, hop);
    std::vector<double> f0(Tn, f);
    Matrix<double> two(2, Tn, 0.3), four(4, Tn, 0.3);
    auto y2 = synth.forward(f0, two, n);
    auto y4 = synth.forward(f0, four, n);
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, std::fabs(y2[i] - y4[i]));
    check(worst == 0.0, "harmonics above Nyquist contribute nothing");
}

void test_harmonic_gradient() {
    std::cout << "HarmonicSynth, gradient (double precision):\n";
    const double sr = 8000.0;
    const std::size_t hop = 16, n = 150, K = 5;
    const std::size_t Tn = HarmonicSynth<double>::frames_for(n, hop);
    HarmonicSynth<double> synth(sr, hop);
    std::vector<double> f0(Tn);
    for (std::size_t t = 0; t < Tn; ++t) f0[t] = 300.0 + 15.0 * t;   // a glide
    Matrix<double> A(K, Tn);
    A.randomizeUniform(0.0, 1.0, 7);
    const auto wt = rand_vec(n, 8);                  // loss = sum(wt .* y)

    synth.forward(f0, A, n);
    auto dA = synth.backward(wt);

    const double eps = 1e-6;
    double worst = 0.0;
    for (std::size_t t = 0; t < Tn; ++t)
        for (std::size_t k = 0; k < K; ++k) {
            auto Ap = A, Am = A;
            Ap(k, t) += eps; Am(k, t) -= eps;
            const double num = (dot(wt, synth.forward(f0, Ap, n))
                              - dot(wt, synth.forward(f0, Am, n))) / (2 * eps);
            worst = std::max(worst, std::fabs(num - dA(k, t)));
        }
    std::cout << "  worst |num - analytic| for dA = " << worst << "\n";
    check(worst < 1e-8, "dA matches numerical");
}

// ---- FilteredNoise -------------------------------------------------------

// All gains 1: the filter does nothing, so the output must be the cached
// noise itself, which has the variance of uniform noise on [-1, 1] (1/3).
void test_noise_identity() {
    std::cout << "FilteredNoise, unit gains pass the noise through:\n";
    const std::size_t hop = 64, B = 9, n = 4096;
    const std::size_t Tn = (n + hop - 1) / hop + 1;
    FilteredNoise<double> noise(hop, B, 3);
    Matrix<double> M(B, Tn, 1.0);
    auto y = noise.forward(M, n);
    double var = 0.0;
    for (std::size_t i = hop; i < n - hop; ++i) var += y[i] * y[i];
    var /= double(n - 2 * hop);
    std::cout << "  variance = " << var << " (expected about 0.333)\n";
    check(std::fabs(var - 1.0 / 3.0) < 0.03, "output variance is that of the noise");

    Matrix<double> Z(B, Tn, 0.0);
    auto z = noise.forward(Z, n);
    double peak = 0.0;
    for (double v : z) peak = std::max(peak, std::fabs(v));
    check(peak == 0.0, "zero gains give silence");
}

// A low-pass setting of the gains must remove high-frequency energy.
void test_noise_shapes_spectrum() {
    std::cout << "FilteredNoise, gains shape the spectrum:\n";
    const std::size_t hop = 128, B = 5, n = 8192;
    const std::size_t Tn = (n + hop - 1) / hop + 1;
    FilteredNoise<double> noise(hop, B, 4);
    Matrix<double> M(B, Tn, 0.0);
    for (std::size_t t = 0; t < Tn; ++t) { M(0, t) = 1.0; M(1, t) = 1.0; }  // low bands only
    auto y = noise.forward(M, n);

    std::vector<std::complex<double>> buf(4096);
    for (std::size_t i = 0; i < 4096; ++i) buf[i] = y[1024 + i];
    fft(buf);
    double low = 0.0, high = 0.0;
    for (std::size_t k = 1; k < 2048; ++k) {
        const double p = std::norm(buf[k]);
        if (k < 400) low += p; else if (k > 1300) high += p;
    }
    std::cout << "  low-band energy / high-band energy = " << low / high << "\n";
    // Not a brick wall: each short frame is windowed, so some energy leaks
    // across bins.  Three orders of magnitude (30 dB) is what to expect.
    check(low > 1e3 * high, "the muted bands carry over 1000x less energy");
}

void test_noise_gradient() {
    std::cout << "FilteredNoise, gradient (double precision):\n";
    const std::size_t hop = 16, B = 4, n = 100;
    const std::size_t Tn = (n + hop - 1) / hop + 1;
    FilteredNoise<double> noise(hop, B, 5);
    Matrix<double> M(B, Tn);
    M.randomizeUniform(0.0, 1.0, 9);
    const auto wt = rand_vec(n, 10);

    noise.forward(M, n);
    auto dM = noise.backward(wt);

    const double eps = 1e-6;
    double worst = 0.0;
    for (std::size_t t = 0; t < Tn; ++t)
        for (std::size_t b = 0; b < B; ++b) {
            auto Mp = M, Mm = M;
            Mp(b, t) += eps; Mm(b, t) -= eps;
            const double num = (dot(wt, noise.forward(Mp, n))
                              - dot(wt, noise.forward(Mm, n))) / (2 * eps);
            worst = std::max(worst, std::fabs(num - dM(b, t)));
        }
    std::cout << "  worst |num - analytic| for dM = " << worst << "\n";
    check(worst < 1e-8, "dM matches numerical");
}

// ---- SpectralLoss --------------------------------------------------------

void test_loss_basics() {
    std::cout << "SpectralLoss, basic properties:\n";
    const std::size_t n = 1024;
    std::vector<double> x(n), shifted(n), other(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i]       = 0.5 * std::sin(2 * PI * 0.05 * i);
        shifted[i] = 0.5 * std::sin(2 * PI * 0.05 * i + 1.3);   // same tone, new phase
        other[i]   = 0.5 * std::sin(2 * PI * 0.11 * i);         // different pitch
    }
    SpectralLoss<double> loss({256, 64});
    loss.set_target(x);
    const double same = loss.forward(x);
    const double ph   = loss.forward(shifted);
    const double diff = loss.forward(other);
    std::cout << "  loss: identical " << same << "  phase-shifted " << ph
              << "  other pitch " << diff << "\n";
    check(same == 0.0, "loss of a signal against itself is 0");
    // Not exactly zero: the frames at the two ends of the signal see the
    // shifted sine start and stop at a different point of its cycle.
    check(ph < 0.1 * diff, "a phase shift costs over 10x less than a pitch change");

    // The waveform difference, by contrast, is large for the phase shift.
    double wave_ph = 0.0, wave_diff = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        wave_ph   += (x[i] - shifted[i]) * (x[i] - shifted[i]);
        wave_diff += (x[i] - other[i])   * (x[i] - other[i]);
    }
    check(wave_ph > 0.5 * wave_diff,
          "sample-wise error cannot tell the two cases apart (sanity)");
}

void test_loss_gradient() {
    std::cout << "SpectralLoss, gradient (double precision):\n";
    const std::size_t n = 200;
    const auto x = rand_vec(n, 20);
    const auto y = rand_vec(n, 21);
    SpectralLoss<double> loss({64, 32, 16});
    loss.set_target(x);
    loss.forward(y);
    const auto dy = loss.backward();

    const double eps = 1e-7;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        auto yp = y, ym = y;
        yp[i] += eps; ym[i] -= eps;
        const double num = (loss.forward(yp) - loss.forward(ym)) / (2 * eps);
        worst = std::max(worst, std::fabs(num - dy[i]));
    }
    std::cout << "  worst |num - analytic| for dy = " << worst << "\n";
    check(worst < 1e-6, "dy matches numerical");
}

// ---- The three together --------------------------------------------------

// Synthesiser -> loss, gradient w.r.t. the harmonic amplitudes: this is
// the exact chain the fitting example and the network training use.
void test_chain_gradient() {
    std::cout << "synth -> loss chain, gradient (double precision):\n";
    const double sr = 8000.0;
    const std::size_t hop = 16, n = 160, K = 4, B = 3;
    const std::size_t Tn = HarmonicSynth<double>::frames_for(n, hop);
    HarmonicSynth<double> synth(sr, hop);
    FilteredNoise<double> noise(hop, B, 6);
    std::vector<double> f0(Tn, 440.0);
    Matrix<double> A(K, Tn), M(B, Tn);
    A.randomizeUniform(0.05, 0.5, 30);
    M.randomizeUniform(0.05, 0.5, 31);
    const auto target = rand_vec(n, 32, -0.5, 0.5);

    SpectralLoss<double> loss({64, 32});
    loss.set_target(target);

    auto run = [&](const Matrix<double>& a, const Matrix<double>& m) {
        auto y = synth.forward(f0, a, n);
        auto z = noise.forward(m, n);
        for (std::size_t i = 0; i < n; ++i) y[i] += z[i];
        return loss.forward(y);
    };

    run(A, M);
    const auto dy = loss.backward();
    auto dA = synth.backward(dy);
    auto dM = noise.backward(dy);

    const double eps = 1e-7;
    double worst_a = 0.0, worst_m = 0.0;
    for (std::size_t t = 0; t < Tn; ++t) {
        for (std::size_t k = 0; k < K; ++k) {
            auto P = A, Q = A; P(k, t) += eps; Q(k, t) -= eps;
            worst_a = std::max(worst_a,
                std::fabs((run(P, M) - run(Q, M)) / (2 * eps) - dA(k, t)));
        }
        for (std::size_t b = 0; b < B; ++b) {
            auto P = M, Q = M; P(b, t) += eps; Q(b, t) -= eps;
            worst_m = std::max(worst_m,
                std::fabs((run(A, P) - run(A, Q)) / (2 * eps) - dM(b, t)));
        }
    }
    std::cout << "  worst |num - analytic|: dA " << worst_a << "  dM " << worst_m << "\n";
    check(worst_a < 1e-6, "dA through the loss matches numerical");
    check(worst_m < 1e-6, "dM through the loss matches numerical");
}

// ---- Pitch ---------------------------------------------------------------

void test_pitch() {
    std::cout << "note names and f0 refinement:\n";
    check(std::fabs(note_to_hz("A4") - 440.0) < 1e-9, "A4 = 440 Hz");
    check(std::fabs(note_to_hz("C4") - 261.6255653) < 1e-6, "C4 = 261.63 Hz");
    check(std::fabs(note_to_hz("A#3") - note_to_hz("Bb3")) < 1e-9, "A#3 = Bb3");
    check(std::fabs(note_to_hz("A2") - 110.0) < 1e-9, "A2 = 110 Hz");

    // A tone 23 cents sharp of A4 with five harmonics; the refinement,
    // told only "A4", must find the true fundamental.
    const double sr = 44100.0;
    const double f_true = 440.0 * std::pow(2.0, 23.0 / 1200.0);
    const std::size_t n = 44100, hop = 512;
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (int h = 1; h <= 5; ++h)
            x[i] += (0.5 / h) * std::sin(2 * PI * f_true * h * i / sr);
    const std::size_t Tn = n / hop + 1;
    auto f0 = refine_f0<double>(x, sr, 440.0, hop, Tn);
    double worst_cents = 0.0;
    for (std::size_t t = 4; t + 4 < Tn; ++t)
        worst_cents = std::max(worst_cents,
                               std::fabs(1200.0 * std::log2(f0[t] / f_true)));
    std::cout << "  worst error = " << worst_cents << " cents\n";
    check(worst_cents <= 1.0, "refined f0 is within 1 cent of the truth");
}

int main() {
    std::cout << "test_synth\n";
    test_harmonic_pure_sine();
    test_harmonic_nyquist();
    test_harmonic_gradient();
    test_noise_identity();
    test_noise_shapes_spectrum();
    test_noise_gradient();
    test_loss_basics();
    test_loss_gradient();
    test_chain_gradient();
    test_pitch();
    std::cout << n_passed << " passed, " << n_failed << " failed\n";
    return n_failed == 0 ? 0 : 1;
}
