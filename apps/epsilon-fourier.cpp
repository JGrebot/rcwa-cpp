#include <cmath>
#include <complex>
#include <utility>

#include <fftw3.h>
#include <Eigen/Dense>

enum class SuppressOption { Normal, SuppressZero };
enum class InPlaneAxis { X, Y };

using Vector2i = std::array<int, 2>;
using Vector2d = std::array<double, 2>;
using Complex = std::complex<double>;

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

template <typename Function>
static Eigen::MatrixXcd Fft2DMatrix(UniformSampler<2> sampler, Function fn) {
    const int n1 = sampler.Size(0), n2 = sampler.Size(1);
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n1 * n2);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n1 * n2);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            const double x = sampler.SamplingPoint<0>(i), y = sampler.SamplingPoint<1>(j);
            const Complex z = fn(x, y);
            fftw_complex *eps = &in[i * n2 + j];
            (*eps)[0] = z.real();
            (*eps)[1] = z.imag();
        }
    }

    fftw_plan p = fftw_plan_dft_2d(n1, n2, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    Eigen::MatrixXcd m(n1, n2);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            fftw_complex *eps = &out[i * n2 + j];
            m(i, j) = Complex((*eps)[0], (*eps)[1]);
        }
    }
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return m;
}

template <typename Matrix, typename Function>
static void WriteMatrixData(const char *filename, const Matrix& m, Function fn, SuppressOption suppress = SuppressOption::Normal) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "cannot open file \"%s\".\n", filename);
        return;
    }
    const int n1 = m.rows(), n2 = m.cols();
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            const double norm = fn(m(i, j));
            fprintf(f, " %g", i == 0 && j == 0 && suppress == SuppressOption::SuppressZero ? 0 : norm);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}


class IndexOrdering2DRowMajor {
public:
    IndexOrdering2DRowMajor(int n1, int n2): n1_(n1), n2_(n2) {}
    Vector2i Unpack(int i) const { return {i / n2_, i % n2_}; }
    int Size() const { return n1_ * n2_; }
private:
    int n1_, n2_;
};

class IndexOrdering2DColMajor {
public:
    IndexOrdering2DColMajor(int n1, int n2): n1_(n1), n2_(n2) {}
    Vector2i Unpack(int i) const { return {i % n1_, i / n1_}; }
    int Size() const { return n1_ * n2_; }
private:
    int n1_, n2_;
};

template <typename Matrix, typename IndexOrdering>
Matrix FftConvolutionMatrix(const int N, const Matrix& m_plain, const IndexOrdering& order) {
    const int n1 = m_plain.rows(), n2 = m_plain.cols();
    const int n_conv = order.Size();
    Matrix m_conv(n_conv, n_conv);
    const double fft_norm_coeff = 1.0 / double(n1 * n2);
    for (int i = 0; i < n_conv; i++) {
        for (int j = 0; j < n_conv; j++) {
            // The components of v1 and v2 below are between 0 and 2*N + 1
            // so we should normally substract N to be between -N and N but
            // since we are going to take the difference between v1 and v2
            // we can omit to substract N.
            const Vector2i v1 = order.Unpack(i);
            const Vector2i v2 = order.Unpack(j);

            // for vdiffx and y we take the negative value because the FFT
            // compute with e^{-i j k / N} where the Whittaker article want
            // the transform with e^{+i j k / N}.
            Vector2i vdiff{-(v1[0] - v2[0]), -(v1[1] - v2[1])};
            if (vdiff[0] < 0) {
                vdiff[0] += n1;
            }
            if (vdiff[1] < 0) {
                vdiff[1] += n2;
            }
            m_conv(i, j) = fft_norm_coeff * m_plain(vdiff[0], vdiff[1]);
        }
    }
    return m_conv;
}

template <typename Function>
Eigen::MatrixXcd KMatrix(InPlaneAxis axis, int n1, int n2, Function k_vector_function) {
    Eigen::MatrixXcd k(n1, n2);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            k(i, j) = (i == j ? k_vector_function(axis, i) : 0);
        }
    }
    return k;
}

template <typename Function>
Eigen::MatrixXcd KKMatrix(InPlaneAxis axis1, InPlaneAxis axis2, int n1, int n2, Function k_vector_function) {
    Eigen::MatrixXcd kk = Eigen::MatrixXcd::Zero(n1, n2);
    for (int i = 0; i < n1; i++) {
        kk(i, i) = k_vector_function(axis1, i) * k_vector_function(axis2, i);
    }
    return kk;
}

template <typename Function>
Eigen::MatrixXcd KEtaKMatrix(InPlaneAxis axis1, const Eigen::MatrixXcd& eta, InPlaneAxis axis2, Function k_vector_function) {
    const int n1 = eta.rows(), n2 = eta.cols();
    Eigen::MatrixXcd k1 = KMatrix(axis1, n1, n2, k_vector_function);
    Eigen::MatrixXcd k2 = KMatrix(axis2, n1, n2, k_vector_function);
    Eigen::MatrixXcd k_eta_k = k1 * eta * k2;
    return k_eta_k;
}

template <typename Function>
Eigen::MatrixXcd FullQEigenValueMatrix(const Eigen::MatrixXcd& epsilon, const Eigen::MatrixXcd& eta, double omega, Function k_vector_function) {
    const int n1 = epsilon.rows(), n2 = epsilon.cols();
    Eigen::MatrixXcd kx = KMatrix(InPlaneAxis::X, n1, n2, k_vector_function);
    Eigen::MatrixXcd ky = KMatrix(InPlaneAxis::Y, n1, n2, k_vector_function);

    Eigen::MatrixXcd wsq_minus_K_script(2 * n1, 2 * n2);
    wsq_minus_K_script.block(0,  0,  n1, n2) = -ky * eta * ky;
    wsq_minus_K_script.block(0,  n2, n1, n2) =  ky * eta * kx;
    wsq_minus_K_script.block(n1, 0,  n1, n2) =  kx * eta * ky;
    wsq_minus_K_script.block(n1, n2, n1, n2) = -kx * eta * kx;
    for (int i = 0; i < 2 * n1; i++) {
        wsq_minus_K_script(i, i) += omega * omega;
    }

    Eigen::MatrixXcd m(2 * n1, 2 * n2);
    m.block(0,  0,  n1, n2) = epsilon;
    m.block(0,  n2, n1, n2) = Eigen::MatrixXcd::Zero(n1, n2);
    m.block(n1, 0,  n1, n2) = Eigen::MatrixXcd::Zero(n1, n2);
    m.block(n1, n2, n1, n2) = epsilon;
    m *= wsq_minus_K_script;

    Eigen::MatrixXcd kxkx = KKMatrix(InPlaneAxis::X, InPlaneAxis::X, n1, n2, k_vector_function);
    Eigen::MatrixXcd kxky = KKMatrix(InPlaneAxis::X, InPlaneAxis::Y, n1, n2, k_vector_function);
    Eigen::MatrixXcd kyky = KKMatrix(InPlaneAxis::Y, InPlaneAxis::Y, n1, n2, k_vector_function);
    m.block(0,  0,  n1, n2) -= kxkx;
    m.block(0,  n2, n1, n2) -= kxky;
    m.block(n1, 0,  n1, n2) -= kxky;
    m.block(n1, n2, n1, n2) -= kyky;

    return m;
}

int main() {
    UniformSampler<2> sampler({2000.0, 2000.0}, {16 * 16, 16 * 16});

    auto epsilon_step_fn = [](double x, double y) -> Complex {
        return (x < 250.0 && y < 250.0 ? 3.55 : 1.0);
    };

    Eigen::MatrixXcd eps = Fft2DMatrix(sampler, epsilon_step_fn);
    Eigen::MatrixXcd eta = Fft2DMatrix(sampler, [&](double x, double y) { return Complex(1,0) / epsilon_step_fn(x, y); });
    WriteMatrixData("output/epsilon-fft.txt", eps, [](Complex m) { return std::real(m); }, SuppressOption::SuppressZero);
    WriteMatrixData("output/eta-fft.txt", eta, [](Complex m) { return std::real(m); }, SuppressOption::SuppressZero);

    const int N = 13; // MAX ORDER
    IndexOrdering2DColMajor reciprocal_space_ordering{2 * N + 1, 2 * N + 1};
    Eigen::MatrixXcd eps_conv = FftConvolutionMatrix(N, eps, reciprocal_space_ordering);
    Eigen::MatrixXcd eta_conv = FftConvolutionMatrix(N, eta, reciprocal_space_ordering);
    WriteMatrixData("output/epsilon-conv.txt", eps_conv, [](Complex m) { return std::real(m); });
    Eigen::MatrixXcd eps_eta_product = eps_conv * eta_conv;
    WriteMatrixData("output/epsilon-eta-product.txt", eps_eta_product, [](Complex z) { return std::real(z); });
    const double pi = 3.14159265358979323846;
    Vector2d k_bloch{2.13 * 2 * pi / sampler.Length(0), 1.3 * 2 * pi / sampler.Length(1)};
    auto bloch_k_vector_function = [&](InPlaneAxis axis, int i_conv) -> double {
        const Vector2i v = reciprocal_space_ordering.Unpack(i_conv);
        const int dim = (axis == InPlaneAxis::X ? 0 : 1);
        return k_bloch[dim] + 2 * pi * v[dim] / sampler.Length(dim);
    };

    const double omega_test = 2.91;
    Eigen::MatrixXcd Q = FullQEigenValueMatrix(eps_conv, eta_conv, omega_test, bloch_k_vector_function);
    WriteMatrixData("output/q-matrix.txt", Q, [](Complex z) { return std::abs(z); });
    return 0;
}
