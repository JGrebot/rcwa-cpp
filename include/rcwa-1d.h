#include <complex>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

enum class Polarization { TE, TM };
enum class PlanarWaveNormalization { UnitaryEField };

struct RcwaConfiguration {
    Polarization polarization;
    double k0;
    double k_bloch;
    int N; // Max order for Fourier transform.
    double L; // Size of the elementary cell.

    // Used to choose the sampling in x when performing FFT of the
    // epsilon and also backward FFT. Very minimal value is 2. Bigger
    // is better for more accuracy.
    int fft_approx_factor;

    int FTSize() const { return 2 * N + 1; }
};

class LayersStack {
public:
    using Function = std::function<std::complex<double>(double)>;
    using Variant = std::variant<std::complex<double>, Function>;
    enum class LayerType { Homogeneous, Patterned };

    template <typename T>
    void AddLayer(double d, T epsilon) {
        layers_.push_back({d, Variant(epsilon)});
    }

    int LayersNumber() const { return layers_.size(); }
    double Thickness(int i) const { return layers_[i].first; }
    Variant Epsilon(int i) const { return layers_[i].second; }

    static LayerType GetLayerType(Variant epsilon) {
        return (epsilon.index() == 0 ? LayerType::Homogeneous : LayerType::Patterned);
    }
private:
    // Thickness and epsilon function of each layer, including the
    // environment and the substrate.
    std::vector<std::pair<double, Variant>> layers_;
};

struct LayerEigenmodes {
    Eigen::VectorXcd Q; // Vector with q_n eigenvalues
    // V is the matrix whose columns are the eigenvectors of the in-plane matrix
    // of the Helmotz equation for the H (magnetic) field in the Fourier space.
    // W is the associated matrix for the electric field, E.
    Eigen::MatrixXcd V, W;
    Eigen::VectorXcd eps_fft, eta_fft; // Vectors with epsilon and 1/epsilon Fourier's transform.
};

struct ScatteringMatrix {
    // The matrices below are the scattering matrices:
    // S1 -> Scattering Matrix S11(0, layer)
    // S2 -> Scattering Matrix S12(0, layer)
    // S3 -> Scattering Matrix S21(layer, N)
    // S4 -> Scattering Matrix S22(layer, N)
    Eigen::MatrixXcd S1, S2, S3, S4;
};

struct RcwaWorkspace {
    using Complex = std::complex<double>;
    using Matrix = Eigen::MatrixXcd;
    using Vector = Eigen::VectorXcd;

    RcwaConfiguration config;
    LayersStack stack;
    std::vector<LayerEigenmodes> eigenmodes;
    std::vector<ScatteringMatrix> scattering_matrix;
    std::vector<Vector> as, bs;
    std::vector<double> flux;

    void ComputeEigenmodes();
    void ComputeScatteringMatrices();
    Vector GetA0Vector(PlanarWaveNormalization normalization);
    void ComputeLayersCoefficients(const Vector& a0);
    void ComputeFluxByLayers();
};

void WriteEMField(const std::string& filename, const RcwaWorkspace& rcwa, const double delta_z);

