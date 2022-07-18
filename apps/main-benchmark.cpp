#include <cmath>
#include <fstream>
#include <string>

#include <fmt/core.h>
#include <Eigen/Dense>

#include "CubicSplineInterpolation.h"
#include "ParseMatrix.h"
#include "rcwa-1d.h"
#include "si_green.h"
#include "write-utils.h"

using Complex = RcwaWorkspace::Complex;
using Matrix = RcwaWorkspace::Matrix;
using Vector = RcwaWorkspace::Vector;

const double pi = 3.14159265358979323846;

using namespace Eigen;

// Wavelengths in nanometers.
CubicSplineInterpolation<ArrayXd, ArrayXcd> ParseSiGreen() {
    ArrayXd wls;
    ArrayXcd n;
    ParseTable(si_green_table,
        [&] (int rows, int cols) {
            wls.resize(rows);
            n.resize(rows);
        },
        [&] (int i, int j, double value) {
            if (j == 0) wls(i) = value;
            if (j == 1) n(i).real(value);
            if (j == 3) n(i).imag(value);
        });
    return CubicSplineInterpolation<ArrayXd, ArrayXcd>{std::move(wls), n * n};
}

int main() {
    auto si_green = ParseSiGreen();
    ArrayXd om = ArrayXd::LinSpaced(50, 1e3*1.24/1200, 1e3*1.24/850);

    const double L = 0.5, theta = 0.01 * pi / 180;

    MatrixXd result(om.size(), 3);
    for (int i = 0; i < om.size(); i++) {
        fmt::print("R and T Calculation for wl = {}\n", 1e3 * 1.24 / om(i));

        const double lambda = 1.24 / om(i); // um
        const double k0 =  2 * pi / lambda;

        const Complex eps_si = si_green(lambda * 1000);
        const Complex eps_sio2 = 2.1054;

        RcwaWorkspace rcwa;
        rcwa.config.polarization = Polarization::TE;
        rcwa.config.N = 100;
        rcwa.config.L = L; // period in um.
        rcwa.config.fft_approx_factor = 8;
        rcwa.config.k0 = k0;
        rcwa.config.k_bloch = k0 * std::sin(theta);

        rcwa.stack.AddLayer(2.5, eps_sio2);
        rcwa.stack.AddLayer(0.25, [=](double x) -> Complex { return (x < 0.2 ? eps_sio2 : eps_si); });
        rcwa.stack.AddLayer(2.5, eps_si);

        rcwa.ComputeEigenmodes();
        rcwa.ComputeScatteringMatrices();

        const int layers_number = rcwa.stack.LayersNumber();

        RcwaWorkspace::Vector a0 = rcwa.GetA0Vector(PlanarWaveNormalization::UnitaryEField);
        rcwa.ComputeLayersCoefficients(a0);
        rcwa.ComputeFluxByLayers();
        result(i, 0) = 1e3 * lambda;
        result(i, 1) = 1 - rcwa.flux[1];
        result(i, 2) = rcwa.flux[layers_number - 1];
        fmt::print("R: {:g} {:g}\n", result(i, 1), result(i, 2));
    }
    std::ofstream f("output/benchmark-result.txt", std::ios::binary);
    auto value_as_is = [](double x) { return x; };
    WriteVectorAsRow(f, result.col(0), value_as_is);
    WriteVectorAsRow(f, result.col(1), value_as_is);
    WriteVectorAsRow(f, result.col(2), value_as_is);
    return 0;
}

