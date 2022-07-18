
template <typename VectorTypeX, typename VectorTypeY>
class LinearInterpolation {
public:
    using ScalarTypeX = typename VectorTypeX::Scalar;
    using ScalarTypeY = typename VectorTypeY::Scalar;

    LinearInterpolation(VectorTypeX x, VectorTypeY y): x_(std::move(x)), y_(std::move(y)) { }

    ScalarTypeY operator()(ScalarTypeX x) const {
        if(x < x_(0) || x > x_[x_.size() - 1])
          return 0;
        int i;
        for (i = 1; i < x_.size(); i++) {
            if (x_(i) >= x) break;
        }
        return y_(i - 1) + (y_(i) - y_(i - 1)) * (x - x_(i - 1)) / (x_(i) - x_(i - 1));
    }
private:
    VectorTypeX x_;
    VectorTypeY y_;
};

