#include <cmath>

/* fmt should be always included before Eigen headers because these latters
   may include <complex.h> from lapacke and it defines I that conflict with
   the fmt library. An alternative solution is to use LAPACK_COMPLEX_CUSTOM
   to prevent including <complex.h> */
#include <fmt/core.h>

#include "rcwa-1d.h"
#include "write-utils.h"

using Complex = RcwaWorkspace::Complex;
using Matrix = RcwaWorkspace::Matrix;
using Vector = RcwaWorkspace::Vector;

const double pi = 3.14159265358979323846;

int main() {
    const double k0 = 1.0, L = 20.0, theta = 60 * pi / 180;
    const double graph_delta_z = 0.2;

    RcwaWorkspace rcwa;
    rcwa.config.polarization = Polarization::TM;
    rcwa.config.k0 = k0;
    rcwa.config.N = 19;
    rcwa.config.L = L;
    rcwa.config.fft_approx_factor = 8;
    rcwa.config.k_bloch = k0 * std::sin(theta);

    const Complex eps_env = 1.0;
    const Complex eps_sub = Complex(12.96, 0.0072);

    rcwa.stack.AddLayer(30.0, eps_env);
    rcwa.stack.AddLayer(10.0, [=](double x) -> Complex { return (x < 0.25 * L ? eps_sub : eps_env); });
    rcwa.stack.AddLayer(45.0, eps_sub);

    rcwa.ComputeEigenmodes();
    rcwa.ComputeScatteringMatrices();

    const int layers_number = rcwa.stack.LayersNumber();
    WriteComplexMatrixData("output/S1-matrix.txt", rcwa.scattering_matrix[layers_number - 1].S1);
    WriteComplexMatrixData("output/S3-matrix.txt", rcwa.scattering_matrix[0].S3);

    Vector a0 = rcwa.GetA0Vector(PlanarWaveNormalization::UnitaryEField);
    rcwa.ComputeLayersCoefficients(a0);

    WriteComplexMatrixData("output/b0-vector.txt", rcwa.bs[0]);
    WriteComplexMatrixData("output/an-vector.txt", rcwa.as[layers_number - 1]);

    WriteEMField("output/h-field.txt", rcwa, graph_delta_z);
    rcwa.ComputeFluxByLayers();

#if 0
    for (int l = 0; l < layers_number - 1; l++) {
        Vector h_boundary_r = ComputeHField(config, stack.Thickness(l), eigenmodes[l], as[l], bs[l], stack.Thickness(l));
        Vector h_boundary_l = ComputeHField(config, stack.Thickness(l + 1), eigenmodes[l + 1], as[l + 1], bs[l + 1], 0.0);
        char x_filename[256];
        sprintf(x_filename, "output/h-boundary-%d%dr.txt", l, l+1);
        WriteComplexMatrixData(x_filename, h_boundary_r);
        sprintf(x_filename, "output/h-boundary-%d%dl.txt", l, l+1);
        WriteComplexMatrixData(x_filename, h_boundary_l);
    }
#endif
    return 0;
}

