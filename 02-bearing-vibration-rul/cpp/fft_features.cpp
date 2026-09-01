// Hot-path C++ de extraccion de features espectrales: FFT radix-2
// Cooley-Tukey escrita a mano (sin FFTW ni ninguna dependencia externa --
// mismo estilo "cero dependencias" de market-tick-anomaly-engine-cpp en
// este portafolio), pensado para un dispositivo de monitoreo embebido real
// donde Python/SciPy no son una opcion.
//
// Verificado contra Python (scipy.fft.rfft via src/features.py) sobre los
// PRIMEROS 16384 puntos (2^14, la potencia de 2 mas cercana a los 20.480
// originales) de 5 snapshots reales -- se trunca en AMBOS lados de forma
// identica, para que la comparacion sea justa (SciPy soporta FFT de
// longitud arbitraria; esta implementacion radix-2 no, por diseno).
//
//   cl /EHsc /O2 /std:c++17 fft_features.cpp /Fe:fft_features.exe
//   fft_features.exe

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

using cd = std::complex<double>;
constexpr double PI = 3.14159265358979323846;
constexpr size_t N = 16384; // 2^14
constexpr double SAMPLING_RATE_HZ = 20000.0;

// FFT iterativa in-place, Cooley-Tukey radix-2 (requiere N potencia de 2).
void fft(std::vector<cd>& a) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2 * PI / static_cast<double>(len);
        cd wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1);
            for (size_t k = 0; k < len / 2; ++k) {
                cd u = a[i + k];
                cd v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

struct Features {
    double rms;
    double kurtosis;
    double dominant_frequency_hz;
    double spectral_entropy;
    double energy_2000_5000hz;
};

Features extract_features(const std::vector<double>& signal) {
    double mean = std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size();

    double sum_sq = 0.0, sum4 = 0.0;
    for (double x : signal) {
        double d = x - mean;
        sum_sq += d * d;
        sum4 += d * d * d * d;
    }
    double variance = sum_sq / signal.size();
    double std_dev = std::sqrt(variance);

    double rms = std::sqrt(std::accumulate(signal.begin(), signal.end(), 0.0,
        [](double acc, double x) { return acc + x * x; }) / signal.size());

    // Kurtosis de Fisher (exceso), igual que scipy.stats.kurtosis por defecto.
    double kurtosis = (sum4 / signal.size()) / (variance * variance) - 3.0;

    // FFT sobre la señal centrada (misma convencion que features.py: resta la media antes de la FFT).
    std::vector<cd> spectrum(N);
    for (size_t i = 0; i < N; ++i) spectrum[i] = cd(signal[i] - mean, 0.0);
    fft(spectrum);

    size_t n_freqs = N / 2 + 1; // rfft: solo mitad positiva del espectro
    std::vector<double> magnitude(n_freqs);
    for (size_t i = 0; i < n_freqs; ++i) magnitude[i] = std::abs(spectrum[i]);

    // Frecuencia dominante (excluye DC, igual que features.py).
    size_t dominant_idx = 1;
    for (size_t i = 2; i < n_freqs; ++i) {
        if (magnitude[i] > magnitude[dominant_idx]) dominant_idx = i;
    }
    double freq_resolution = SAMPLING_RATE_HZ / static_cast<double>(N);
    double dominant_frequency_hz = dominant_idx * freq_resolution;

    // Entropia espectral: -sum(p*log(p)) sobre el PSD normalizado.
    double total_energy = 0.0;
    for (double m : magnitude) total_energy += m * m;
    double entropy = 0.0;
    for (double m : magnitude) {
        double p = (m * m) / (total_energy + 1e-12) + 1e-12;
        entropy -= p * std::log(p);
    }

    // Energia en banda 2000-5000Hz, normalizada por energia total.
    double band_energy = 0.0;
    for (size_t i = 0; i < n_freqs; ++i) {
        double f = i * freq_resolution;
        if (f >= 2000.0 && f < 5000.0) band_energy += magnitude[i] * magnitude[i];
    }
    double energy_2000_5000hz = band_energy / (total_energy + 1e-12);

    return {rms, kurtosis, dominant_frequency_hz, entropy, energy_2000_5000hz};
}

std::vector<double> load_signal(const std::string& path) {
    std::vector<double> signal;
    signal.reserve(N);
    std::ifstream file(path);
    double value;
    while (file >> value && signal.size() < N) signal.push_back(value);
    return signal;
}

int main() {
    std::cout << "[1/2] Cargando snapshots reales y verificando contra Python...\n";

    std::ifstream ref_file("python_reference.csv");
    std::string line;
    std::getline(ref_file, line); // header

    double max_abs_diff = 0.0;
    std::vector<std::vector<double>> loaded_signals;

    while (std::getline(ref_file, line)) {
        std::stringstream ss(line);
        std::string filename;
        std::getline(ss, filename, ',');

        std::string val;
        std::vector<double> expected;
        while (std::getline(ss, val, ',')) expected.push_back(std::stod(val));

        auto signal = load_signal("snapshot_" + filename + ".txt");
        loaded_signals.push_back(signal);
        Features f = extract_features(signal);
        double computed[5] = {f.rms, f.kurtosis, f.dominant_frequency_hz, f.spectral_entropy, f.energy_2000_5000hz};

        std::cout << "  " << filename << ": ";
        for (int i = 0; i < 5; ++i) {
            double diff = std::abs(computed[i] - expected[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
        }
        std::cout << "rms=" << f.rms << " kurtosis=" << f.kurtosis
                   << " dom_freq=" << f.dominant_frequency_hz
                   << " entropy=" << f.spectral_entropy
                   << " band_energy=" << f.energy_2000_5000hz << "\n";
    }
    std::cout << "  Diferencia absoluta maxima vs. Python: " << max_abs_diff << "\n";

    std::cout << "\n[2/2] Benchmark de latencia (FFT + features, 1000 repeticiones sobre snapshots reales)...\n";
    auto start = std::chrono::high_resolution_clock::now();
    const int reps = 1000;
    for (int r = 0; r < reps; ++r) {
        for (auto& s : loaded_signals) {
            volatile Features f = extract_features(s); // volatile: evita que el optimizador elimine el calculo
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_snapshot_us = (total_ms * 1000.0) / (reps * loaded_signals.size());

    std::cout << "\n=== Resultado ===\n";
    std::cout << "Snapshots procesados: " << (reps * loaded_signals.size()) << "\n";
    std::cout << "Tiempo total: " << total_ms << "ms\n";
    std::cout << "Latencia por snapshot (FFT de 16384 puntos + 5 features): " << per_snapshot_us << "us\n";
    std::cout << "Throughput: " << (1000000.0 / per_snapshot_us) << " snapshots/segundo\n";
    std::cout << "Diferencia maxima vs. Python: " << max_abs_diff << "\n";

    return 0;
}
