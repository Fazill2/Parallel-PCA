#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Helper function to perform matrix multiplication
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B.size();
    int p = B[0].size();
    vector<vector<double>> C(n, vector<double>(p, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < m; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

double norm(const vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

vector<double> scalarMultiply(const vector<double>& x, double alpha) {
    vector<double> result(x.size(), 0.0);
    for (int i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i];
    }
    return result;
}

vector<vector<double>> identityMatrix(int n) {
    vector<vector<double>> I(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

// hausolder transformation
pair<vector<vector<double>>, vector<vector<double>>> householderTransformation(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> Q = identityMatrix(n);
    vector<vector<double>> R = A;

    for (int i = 0; i < n - 1; ++i) {
        vector<double> x(n - i, 0.0);
        for (int j = i; j < n; ++j) {
            x[j - i] = R[j][i];
        }

        double alpha = norm(x);

        vector<double> e(n - i, 0.0);
        e[0] = 1.0;

        vector<double> v(n - i, 0.0);
        for (int j = 0; j < n - i; ++j) {
            v[j] = x[j] - alpha * e[j];
        }
        vector<double> normVec(n - i, 0.0);
        normVec = scalarMultiply(v, 1.0 / norm(v));
        
        vector<vector<double>> H = identityMatrix(n);
        for (int j = 0; j < n - i; ++j) {
            for (int k = 0; k < n - i; ++k) {
                H[j + i][k + i] -= 2 * normVec[j] * normVec[k];
            }
        }
        Q = matrixMultiply(Q, H);
        R = matrixMultiply(H, R);
    }
    // set the lower triangular matrix to 0
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            R[i][j] = 0.0;
        }
    }
    return make_pair(Q, R);
}



int main() {
    // Example matrix A
    vector<vector<double>> A = {
        {0.5, 0.75, 0.5},
        {1.0,  0.5, 0.75},
        {0.25, 0.25, 0.25}
    };

    vector<vector<double>> A2 = {
        {0.5, 1.0, 0.25},
        {0.75, 0.5, 0.25},
        {0.5, 0.75, 0.25}
    };

    // Set the desired tolerance (epsilon) for convergence
    double epsilon = 1e-6;

    // Perform the Householder transformation
    pair<vector<vector<double>>, vector<vector<double>>> QR = householderTransformation(A);
    vector<vector<double>> Q = QR.first;
    vector<vector<double>> R = QR.second;

    // Output the orthogonal matrix Q
    cout << "Orthogonal matrix Q:" << endl;
    for (int i = 0; i < Q.size(); ++i) {
        for (int j = 0; j < Q[0].size(); ++j) {
            cout << Q[i][j] << " ";
        }
        cout << endl;
    }

    // Output the upper triangular matrix R
    cout << "Upper triangular matrix R:" << endl;
    for (int i = 0; i < R.size(); ++i) {
        for (int j = 0; j < R[0].size(); ++j) {
            cout << R[i][j] << " ";
        }
        cout << endl;
    }


    return 0;
}
