/* Adapted from https://github.com/CD3/libInterpolate

  Original copyright notice:

MIT License

Copyright (c) 2017 C.D. Clark III

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

template <typename VectorTypeX, typename VectorTypeY>
class CubicSplineInterpolation {
public:
    using ScalarTypeX = typename VectorTypeX::Scalar;
    using ScalarTypeY = typename VectorTypeY::Scalar;

    CubicSplineInterpolation(VectorTypeX X_tmp, VectorTypeY Y_tmp): xData_(std::move(X_tmp)), yData_(std::move(Y_tmp)), a_(xData_.size() - 1), b_(xData_.size() - 1) {
        const VectorTypeX& X = xData_;
        const VectorTypeY& Y = yData_;

        /*
         * Solves Ax=B using the Thomas algorithm, because the matrix A will be tridiagonal and diagonally dominant.
         *
         * The method is outlined on the Wikipedia page for Tridiagonal Matrix Algorithm
         */
        const int N = X.size();

        //init the matrices that get solved
        VectorTypeX Aa(N), Ab(N), Ac(N);
        VectorTypeY bb(N);

        //Ac is a vector of the upper diagonals of matrix A
        //
        //Since there is no upper diagonal on the last row, the last value must be zero.
        for (int i = 0; i < N - 1; ++i) {
            Ac(i) = 1 / (X(i+1) - X(i));
        }

        //This is the line that was causing breakage for n=odd. It was Ac(X.size()) and should have been Ac(X.size()-1)
        Ac(N - 1) = 0.0;

        //Ab is a vector of the diagnoals of matrix A
        Ab(0) = 2 / (X(1) - X(0));
        for (int i = 1; i < N - 1; ++i) {
            Ab(i) = 2 / (X(i) - X(i - 1)) + 2 / (X(i + 1) - X(i));
        }
        Ab(N - 1) = 2 / (X(N - 1) - X(N - 1 - 1));

        //Aa is a vector of the lower diagonals of matrix A
        //
        //Since there is no upper diagonal on the first row, the first value must be zero.
        Aa(0) = 0.0;
        for (int i = 1; i < N; ++i) {
            Aa(i) = 1 / (X(i) - X(i - 1));
        }

        // setup RHS vector
        for(int i = 0; i < N; ++i) {
            if (i == 0) {
                bb(i) = 3.0 * (Y(i + 1) - Y(i)) / std::pow(X(i + 1) - X(i), 2);
            } else if (i == N - 1) {
                bb(i) = 3.0 * (Y(i) - Y(i - 1)) / std::pow(X(i) - X(i - 1), 2);
            } else {
                bb(i) = 3.0 * ((Y(i) - Y(i - 1)) / (std::pow(X(i) - X(i - 1), 2)) + (Y(i + 1) - Y(i)) / std::pow(X(i + 1) - X(i), 2));
            }
        }

        VectorTypeX c_star(N);

        c_star(0) = Ac(0) / Ab(0);
        for (int i = 1; i < c_star.size(); ++i) {
            c_star(i) = Ac(i) / (Ab(i) - Aa(i) * c_star(i - 1));
        }

        VectorTypeY d_star(N);
        d_star(0) = bb(0) / Ab(0);

        for (int i = 1; i < N; ++i) {
            d_star(i) = (bb(i) - Aa(i) * d_star(i - 1)) / (Ab(i) - Aa(i)*c_star(i - 1));
        }

        VectorTypeY x(N);
        x(N - 1) = d_star(N - 1);

        for (int i = N - 1; i-- > 0;) {
            x(i) = d_star(i) - c_star(i) * x(i + 1);
        }

        for (int i = 0; i < N - 1; ++i) {
            a_(i) = x(i) * (X(i + 1) - X(i)) - (Y(i + 1) - Y(i));
            b_(i) = -x(i + 1) * (X(i + 1) - X(i)) + (Y(i + 1) - Y(i));
        }
    }

    ScalarTypeY operator()(ScalarTypeX x) const {
        if(x < xData_(0) || x > xData_[xData_.size() - 1])
          return 0;

        const VectorTypeX& X = xData_;
        const VectorTypeY& Y = yData_;

        int i;
        for (i = 1; i < X.size(); i++) {
            if (X(i) >= x) break;
        }

        // See the wikipedia page on "Spline interpolation" (https://en.wikipedia.org/wiki/Spline_interpolation)
        // for a derivation this interpolation.
        ScalarTypeX t = (x - X(i - 1)) / (X(i) - X(i - 1));
        return (1 - t) * Y(i - 1) + t * Y(i) + t * (1 - t) * (a_[i - 1] * (1 - t) + b_[i - 1] * t);
    }

private:
    VectorTypeX xData_;
    VectorTypeY yData_;
    VectorTypeY a_, b_;
};

