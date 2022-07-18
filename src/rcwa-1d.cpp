#include <cmath>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <functional>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

/* fmt should be always included before Eigen headers because these latters
   may include <complex.h> from lapacke and it defines I that conflict with
   the fmt library. An alternative solution is to use LAPACK_COMPLEX_CUSTOM
   to prevent including <complex.h> */
#include <fmt/core.h>

#include <Eigen/Eigenvalues>
#include <fftw3.h>

#include "rcwa-1d.h"
#include "write-utils.h"

using Array2i = std::array<int, 2>;
using Array2d = std::array<double, 2>;
using Complex = std::complex<double>;

using Matrix = Eigen::MatrixXcd;
using Vector = Eigen::VectorXcd;

const double pi = 3.14159265358979323846;

using namespace std::complex_literals;

template <unsigned Rank>
class UniformSampler {
public:
    UniformSampler(const std::array<double, Rank>& ls, const std::array<int, Rank>& ns): l_{ls}, n_{ns} {}

    double Length(int dim) const { return l_[dim]; }
    int Size(int dim) const { return n_[dim]; }

    template <unsigned Dim>
    double SamplingPoint(int index) const {
        return l_[Dim] * index / n_[Dim];
    }
private:
    std::array<double, Rank> l_;
    std::array<int, Rank> n_;
};

class FftWorkspace {
public:
    FftWorkspace(int n, int sign, unsigned flags = FFTW_ESTIMATE): n_(n) {
        in_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        out_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
        p_ = fftw_plan_dft_1d(n, in_, out_, sign, flags);
    }
    ~FftWorkspace() {
        fftw_destroy_plan(p_);
        fftw_free(in_);
        fftw_free(out_);
    }
    void SetInputValue(int i, Complex z) {
        in_[i][0] = z.real();
        in_[i][1] = z.imag();
    }
    Complex GetOutputValue(int i) const {
        return Complex{out_[i][0], out_[i][1]};
    }
    void Transform() {
        fftw_execute(p_);
    }
    int Size() const { return n_; }
private:
    int n_;
    fftw_plan p_;
    fftw_complex *in_, *out_;
};

template <typename Function>
static Vector FftMatrix1D(FftWorkspace& ws, const UniformSampler<1>& sampler, Function fn) {
    const int n = sampler.Size(0);
    for (int i = 0; i < n; i++) {
        const double x = sampler.SamplingPoint<0>(i);
        ws.SetInputValue(i, fn(x));
    }
    ws.Transform();
    Vector v(n);
    for (int i = 0; i < n; i++) {
        v(i) = ws.GetOutputValue(i);
    }
    return v;
}

// Fft transform of a constant function but cutting between -N and N.
// It should be usable to construct the Toeplitz matrix with the function
// FftConvolutionMatrix1D.
Vector FftVectorHomogeneous(const Complex& value, const int N) {
    Vector v = Vector::Zero(2 * N + 1);
    v(0) = double(2 * N + 1) * value;
    return v;
}

static Vector InverseFftVector1D(FftWorkspace& ws, const int N, const double k_bloch, const UniformSampler<1>& sampler, const Vector& vec) {
    const int n = ws.Size();
    for (int i = 0; i < vec.rows(); i++) {
        const Complex z = vec(i);
        const int g = i - N;
        ws.SetInputValue(g >= 0 ? g : n + g, z);
    }

    for (int i = N + 1; i < n - N; i++) {
        ws.SetInputValue(i, 0);
    }

    ws.Transform();

    Vector vec_x(n);
    for (int i = 0; i < n; i++) {
        const Complex z = ws.GetOutputValue(i);
        const double x = sampler.SamplingPoint<0>(i);
        vec_x(i) = std::exp(1i * k_bloch * x) * z;
    }
    return vec_x;
}

// The argument N below correspond to the max order for the fourier transform.
template <typename VectorType, typename MatrixType>
Matrix FftConvolutionMatrix1D(const int N, const VectorType& m_plain) {
    const int n_fft = m_plain.rows();
    const int n_conv = 2 * N + 1;
    MatrixType m_conv(n_conv, n_conv);
    const double fft_norm_coeff = 1.0 / double(n_fft);
    for (int i = -N; i <= N; i++) {
        for (int j = -N; j <= N; j++) {
            // for vdiffx and y we take the negative value because the FFT
            // compute with e^{-i j k / N} where the Whittaker article want
            // the transform with e^{+i j k / N}.
            int i_diff = (i - j);
            if (i_diff < 0) {
                i_diff += n_fft;
            }
            m_conv(i + N, j + N) = fft_norm_coeff * m_plain(i_diff);
        }
    }
    return m_conv;
}

// The argument "factor" should be at least two or more. For a reasonable
// accuracy on well behaved function something between 8 or 16 should be fine.
int EstimateFftSizeForKMax(const double k_max, const double L, const int factor) {
    double n_opt = factor * (k_max * L) / (2 * pi);
    const int log_n = std::ceil(std::log2(n_opt));
    return (1 << log_n);
}

UniformSampler<1> GetOptimalSampler(const RcwaConfiguration& config) {
    const double k_max = std::fabs(config.k_bloch) + 2 * pi * config.N / config.L;
    const int fft_size = EstimateFftSizeForKMax(k_max, config.L, config.fft_approx_factor);
    return {{config.L}, {fft_size}};
}

enum class ReciprocalSpaceOption { Standard, IncludeBlochVector };

Vector ReciprocalSpaceVector1D(const RcwaConfiguration& config, ReciprocalSpaceOption option) {
    const int N = config.N;
    const int Ns = config.FTSize();
    const double dk = 2 * pi / config.L;
    Vector k(Ns);
    const double k_offset = (option == ReciprocalSpaceOption::IncludeBlochVector ? config.k_bloch : 0.0);
    for (int i = 0; i < config.FTSize(); i++) {
        k(i) = k_offset + dk * (i - N);
    }
    return k;
}

Complex SqrtWhittaker(Complex x) {
    const Complex y = std::sqrt(x);
    return (std::imag(y) >= 0 ? y : -y);
}

template <typename VectorType>
VectorType SqrtWhittaker(const VectorType& v) {
    const int n = v.rows();
    VectorType v_sqrt{n};
    for (int i = 0; i < n; i++) {
        v_sqrt(i) = SqrtWhittaker(v(i));
    }
    return v_sqrt;
}

LayerEigenmodes SolveLayerHomogeneous(const RcwaConfiguration& config, Complex epsilon) {
    const int Ns = config.FTSize();
    Vector kx = ReciprocalSpaceVector1D(config, ReciprocalSpaceOption::IncludeBlochVector);
    // Note that in this case (1D planar incidence TM mode) V is the identity
    // matrix and can be omitted.
    // The Whittaker paper use normally the matrix W:
    //
    // W = (w^2 - K_script) * V * q^{-1}
    //
    // In C++ without V:
    // Matrix W = wsq_minus_K_script.asDiagonal() * Matrix{Q.asDiagonal().inverse()};
    //
    // but q^{-1} can have infinite values if q_n = 0 for some n.
    // We use instead:
    //
    // W = eta * V * q = eta * q
    //
    // Note that here we are treating the 1D planar incidence TM mode.
    // For the general case it will be:
    //
    // W = H_script * (K * V * q^{-1} + V * q)
    //
    // where V is denoted like Phi in Whittaker's notation.
    const Complex eta = 1.0 / epsilon;
    Vector wsq_minus_K_script = (std::pow(config.k0, 2) - eta * kx.array() * kx.array()).matrix();
    Vector Q = SqrtWhittaker<Vector>(epsilon * wsq_minus_K_script);
    Matrix V = Matrix::Identity(Ns, Ns);
    Matrix W;
    if (config.polarization == Polarization::TM) {
        // Form (A)
        W = eta * Q.asDiagonal();
        // Alternate form (B):
        // W = wsq_minus_K_script * Q.asDiagonal().inverse();
        // Equivalent to first one but will buf is some q_n is zero.
    } else {
        // The formula for W is W = k0^2 V q^{-1} but V is equal to the identity here.
        // FIXME: solve the problem where some q are to zero.
        // Form (B):
        W = std::pow(config.k0, 2) * Q.asDiagonal().inverse();
        // Alternate form (A) (not verified):
        // W = (eta * (kx.array().square() * Q.array().inverse() + Q.array())).matrix().asDiagonal();
    }
    Vector eps_fft = FftVectorHomogeneous(epsilon, config.N);
    Vector eta_fft = FftVectorHomogeneous(eta, config.N);
    return LayerEigenmodes{std::move(Q), std::move(V), std::move(W), std::move(eps_fft), std::move(eta_fft)};
}

template <typename Function>
LayerEigenmodes SolveLayer1DPatternPlanar(const RcwaConfiguration& config, Function epsilon_fn) {
    UniformSampler<1> sampler = GetOptimalSampler(config);
    FftWorkspace fft_workspace_fw{sampler.Size(0), FFTW_FORWARD};
    Vector eps_fft = FftMatrix1D(fft_workspace_fw, sampler, epsilon_fn);
    Vector eta_fft = FftMatrix1D(fft_workspace_fw, sampler, [&](double x) { return Complex(1,0) / epsilon_fn(x); });

    // Compute the related Toeplitz matrices.
    Matrix eps = FftConvolutionMatrix1D<Vector, Matrix>(config.N, eps_fft);
    Matrix eta = FftConvolutionMatrix1D<Vector, Matrix>(config.N, eta_fft);

    Vector kx = ReciprocalSpaceVector1D(config, ReciprocalSpaceOption::IncludeBlochVector);

    const int Ns = config.FTSize();
    Matrix H, wsq_minus_K_script;
    if (config.polarization == Polarization::TM) {
        // Eigen lu().solve does not accept kx.asDiagonal() hence we explicitly build a Matrix.
        wsq_minus_K_script = std::pow(config.k0, 2) * Matrix::Identity(Ns, Ns) - (kx.asDiagonal() * eps.lu().solve(Matrix{kx.asDiagonal()}));

        // With the product below we compute the matrix:
        // H = (1/eps)^{-1} (k0^2 - kx * eps^{-1} * kx)
        // where (1/eps) = eta (both are Toeplitz matrices).
        // Note that the use of (1/eps)^{-1} and eps^{-1} is mandated for an optimal
        // convergence by:
        // Lifeng Li, "Use of Fourier series in the analysis of discontinuous periodic
        // structures", J. Opt. Soc. Am. A 13, 1870-1876 (1996)
        H = eta.lu().solve(wsq_minus_K_script);
    } else {
        H = Matrix{eps * std::pow(config.k0, 2)} - Matrix{kx.array().square().matrix().asDiagonal()};
    }

    Eigen::ComplexEigenSolver<Matrix> ces;
    ces.compute(H);
    const Vector Q = SqrtWhittaker<Vector>(ces.eigenvalues());
    const Matrix V = ces.eigenvectors();

    // Compute associated eigenvector matrix for the E field.
    Matrix W;
    if (config.polarization == Polarization::TM) {
        W = wsq_minus_K_script * V * Q.asDiagonal().inverse();
    } else {
        W = std::pow(config.k0, 2) * V * Q.asDiagonal().inverse();
    }
    return LayerEigenmodes{std::move(Q), std::move(V), std::move(W), std::move(eps_fft), std::move(eta_fft)};
}

LayerEigenmodes SolveLayer1DPlanar(const RcwaConfiguration& config, LayersStack::Variant epsilon_variant) {
    if (LayersStack::GetLayerType(epsilon_variant) == LayersStack::LayerType::Homogeneous) {
        return SolveLayerHomogeneous(config, std::get<Complex>(epsilon_variant));
    }
    return SolveLayer1DPatternPlanar(config, std::get<LayersStack::Function>(epsilon_variant));
}

std::pair<Matrix, Matrix> ScatteringProductForward(const Matrix& S1, const Matrix& S2, const Matrix& Vi, const Matrix& Vj, const Matrix& Wi, const Matrix& Wj, const Vector& fi, const Vector& fj) {
    const Matrix Vpp = Vi.lu().solve(Vj);
    const Matrix Wpp = Wi.lu().solve(Wj);
    const Matrix Im = 0.5 * (Vpp + Wpp), Jm = 0.5 * (Vpp - Wpp);
    const Eigen::PartialPivLU<Matrix> t1_lu = (Im - fi.asDiagonal() * S2 * Jm).lu();
    Matrix S1r = t1_lu.solve(fi.asDiagonal() * S1);
    Matrix S2r = t1_lu.solve(fi.asDiagonal() * S2 * Im - Jm) * fj.asDiagonal();
    return {std::move(S1r), std::move(S2r)};
}

std::pair<Matrix, Matrix> ScatteringProductBackward(const Matrix& S3, const Matrix& S4, const Matrix& Vi, const Matrix& Vj, const Matrix& Wi, const Matrix& Wj, const Vector& fi, const Vector& fj) {
    const Matrix Vpp = Vi.lu().solve(Vj);
    const Matrix Wpp = Wi.lu().solve(Wj);
    const Matrix Lm = 0.5 * (Vpp + Wpp), Mm = 0.5 * (Vpp - Wpp);
    const Eigen::PartialPivLU<Matrix> t1_lu = (Lm - fi.asDiagonal() * S3 * Mm).lu();
    Matrix S3r = t1_lu.solve(fi.asDiagonal() * S3 * Lm - Mm) * fj.asDiagonal();
    Matrix S4r = t1_lu.solve(fi.asDiagonal() * S4);
    return {std::move(S3r), std::move(S4r)};
}

std::array<Vector, 3> ComputeEMField(const RcwaConfiguration& config, const UniformSampler<1>& sampler, FftWorkspace& fft_workspace, const double d, const LayerEigenmodes& em, const Vector& a, const Vector& b, const double z) {
    const int Ns = 2 * config.N + 1;
    Vector cp(Ns), cm(Ns);
    for (int i = 0; i < Ns; i++) {
        const Complex ffw = std::exp(1i * em.Q(i) * z), fbw = std::exp(1i * em.Q(i) * (d -z));
        cp(i) = ffw * a(i) + fbw * b(i);
        cm(i) = ffw * a(i) - fbw * b(i);
    }
    const Vector kx = ReciprocalSpaceVector1D(config, ReciprocalSpaceOption::IncludeBlochVector);
    const Matrix eta = FftConvolutionMatrix1D<Vector, Matrix>(config.N, em.eta_fft);
    auto RealSpaceVector = [&](const Vector& v) { return InverseFftVector1D(fft_workspace, config.N, config.k_bloch, sampler, v); };
    if (config.polarization == Polarization::TM) {
        Vector h_y = RealSpaceVector(em.V * cp);
        Vector e_x = RealSpaceVector(eta * em.V * em.Q.asDiagonal() * cm);
        Vector e_z = RealSpaceVector(- Matrix{kx.asDiagonal() * em.V} * cp);
        return {std::move(h_y), std::move(e_x), std::move(e_z)};
    }
    Vector h_x = RealSpaceVector(em.V * cp);
    Vector e_y = RealSpaceVector(std::pow(config.k0, 2) * em.V * em.Q.asDiagonal().inverse() * cm);
    Vector h_z = RealSpaceVector(- Matrix{kx.asDiagonal() * em.V * em.Q.asDiagonal().inverse()} * cm);
    return {std::move(e_y), std::move(h_x), std::move(h_z)};
}

void WriteEMField(const std::string& filename, const RcwaWorkspace& rcwa, const double delta_z) {
    const LayersStack& stack = rcwa.stack;
    std::ofstream h_file(filename, std::ios::binary);
    h_file << (rcwa.config.polarization == Polarization::TE ? "PLANAR_1D_TE" : "PLANAR_1D_TM") << std::endl;
    UniformSampler<1> sampler = GetOptimalSampler(rcwa.config);
    FftWorkspace fft_workspace(sampler.Size(0), FFTW_BACKWARD, FFTW_MEASURE);
    const int layers_number = stack.LayersNumber();
    double z_sum = 0.0;
    for (int l = 0; l < layers_number; l++) {
        z_sum += stack.Thickness(l);
    }
    const int x_n = sampler.Size(0), z_n = int(z_sum / delta_z) + 1;
    Eigen::ArrayXd x_array = Eigen::ArrayXd::LinSpaced(x_n, 0.0, sampler.Length(0) * (x_n - 1) / x_n);
    Eigen::ArrayXd z_array = Eigen::ArrayXd::LinSpaced(z_n, 0.0, (z_n - 1) * delta_z);
    WriteVectorAsRow(h_file, x_array, [](double x) { return x; });
    WriteVectorAsRow(h_file, z_array, [&](double z) { return z_sum - z; });
    double prev_z = 0.0, next_z = stack.Thickness(0);
    auto real_part = [](Complex z) { return std::real(z); };
    for (int l = 0, z_i = 0; z_i < z_n; z_i++) {
        double z = z_i * delta_z;
        while (z > next_z) {
            l++;
            prev_z = next_z;
            next_z += stack.Thickness(l);
        }
        auto em_fields = ComputeEMField(rcwa.config, sampler, fft_workspace, stack.Thickness(l), rcwa.eigenmodes[l], rcwa.as[l], rcwa.bs[l], z - prev_z);
        WriteVectorAsRow(h_file, em_fields[0], real_part);
        WriteVectorAsRow(h_file, em_fields[1], real_part);
        WriteVectorAsRow(h_file, em_fields[2], real_part);
    }
}

void RcwaWorkspace::ComputeEigenmodes() {
    const int layers_number = stack.LayersNumber();
    for (int l = 0; l < layers_number; l++) {
        eigenmodes.push_back(SolveLayer1DPlanar(config, stack.Epsilon(l)));
    }
}

void RcwaWorkspace::ComputeScatteringMatrices() {
    const int Ns = config.FTSize();
    const int layers_number = stack.LayersNumber();
    scattering_matrix.resize(layers_number);
    scattering_matrix[0].S1 = Matrix::Identity(Ns, Ns);
    scattering_matrix[0].S2 = Matrix::Zero(Ns, Ns);
    for (int l = 0; l < layers_number - 1; l++) {
        const auto &em1 = eigenmodes[l], &em2 = eigenmodes[l + 1];
        const double d1 = stack.Thickness(l), d2 = stack.Thickness(l + 1);
        Vector f1 = (1i * d1 * em1.Q.array()).exp().matrix();
        Vector f2 = (1i * d2 * em2.Q.array()).exp().matrix();
        std::tie(scattering_matrix[l + 1].S1, scattering_matrix[l + 1].S2) = ScatteringProductForward(scattering_matrix[l].S1, scattering_matrix[l].S2, em1.V, em2.V, em1.W, em2.W, f1, f2);
    }
    scattering_matrix[layers_number - 1].S3 = Matrix::Zero(Ns, Ns);
    scattering_matrix[layers_number - 1].S4 = Matrix::Identity(Ns, Ns);
    for (int l = layers_number - 1; l > 0; l--) {
        const auto &em1 = eigenmodes[l], &em2 = eigenmodes[l - 1];
        const double d1 = stack.Thickness(l), d2 = stack.Thickness(l - 1);
        Vector f1 = (1i * d1 * em1.Q.array()).exp().matrix();
        Vector f2 = (1i * d2 * em2.Q.array()).exp().matrix();
        std::tie(scattering_matrix[l - 1].S3, scattering_matrix[l - 1].S4) = ScatteringProductBackward(scattering_matrix[l].S3, scattering_matrix[l].S4, em1.V, em2.V, em1.W, em2.W, f1, f2);
    }
}

Vector RcwaWorkspace::GetA0Vector(PlanarWaveNormalization normalization) {
    Vector a0 = Vector::Zero(config.FTSize());
    const int N = config.N;
    const Complex epsilon0 = std::get<Complex>(stack.Epsilon(0));
    a0(N) = std::exp(-1i * stack.Thickness(0) * eigenmodes[0].Q(N)) * std::sqrt(epsilon0) / config.k0;
    return a0;
}

void RcwaWorkspace::ComputeLayersCoefficients(const Vector& a0) {
    const int Ns = config.FTSize();
    const int layers_number = stack.LayersNumber();
    as.resize(layers_number);
    bs.resize(layers_number);
    as[0] = a0;
    for (int layer = 0; layer < layers_number; layer++) {
        const auto& sm = scattering_matrix[layer];
        as[layer] = (Matrix::Identity(Ns, Ns) - sm.S2 * sm.S3).lu().solve(sm.S1 * as[0]);
        bs[layer] = (Matrix::Identity(Ns, Ns) - sm.S3 * sm.S2).lu().solve(sm.S3 * sm.S1 * as[0]);
    }
}

void RcwaWorkspace::ComputeFluxByLayers() {
    const double incflux = std::real(eigenmodes[0].Q(config.N)) / std::pow(config.k0, 2);
    const int layers_number = stack.LayersNumber();
    flux.resize(layers_number);
    flux[0] = 1.0;
    for (int l = 1; l < layers_number; l++) {
        const double d = stack.Thickness(l);
        auto& em = eigenmodes[l];
        const Eigen::ArrayXcd fbw = (1i * em.Q.array() * d).exp();
        Vector cp = as[l] + (fbw * bs[l].array()).matrix();
        Vector cm = as[l] - (fbw * bs[l].array()).matrix();
        Vector hy = em.V * cp, ex = em.W * cm;
        flux[l] = std::real(ex.dot(hy)) / incflux;
    }
}

