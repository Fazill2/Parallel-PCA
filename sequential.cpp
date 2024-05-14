#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <functional>
#include <chrono>
#include <iomanip>

using namespace std;
 
// Function to read data from a file
vector<vector<double>> readData(string filename) {
    ifstream file(filename);
    vector<vector<double>> data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        data.push_back(row);
    }
    return data;
}

// Function to calculate the mean of a vector, complexity O(n)
double mean(vector<double> vec) {
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

// Function to calculate standard deviation of a vector, complexity O(n)
double standardDeviation(vector<double> vec) {
    double m = mean(vec);
    double sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += pow(vec[i] - m, 2);
    }
    return sqrt(sum / vec.size());
}

// Function to standardize the vector, complexity O(n)
vector<double> standardize(vector<double> vec) {
    double m = mean(vec);
    double sd = standardDeviation(vec);
    vector<double> standardized;
    for (int i = 0; i < vec.size(); i++) {
        standardized.push_back((vec[i] - m) / sd);
    }
    return standardized;
}

// Function to calculate the covariance of two vectors, complexity O(n)
double covariance(vector<double> vec1, vector<double> vec2) {
    double m1 = mean(vec1);
    double m2 = mean(vec2);
    double sum = 0;
    for (int i = 0; i < vec1.size(); i++) {
        sum += (vec1[i] - m1) * (vec2[i] - m2);
    }
    return sum / vec1.size();
}

// Function to calculate the covariance matrix of a dataset, complexity O(m^2*n), m - columns, n - rows
vector<vector<double>> covarianceMatrix(vector<vector<double>> data) {
    int n = data.size();
    std::vector<std::vector<double>> covMatrix(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            covMatrix[i][j] = covariance(data[i], data[j]);
        }
    }
    return covMatrix;
}
// Helper function to transpose a matrix, complexity O(n*m), n - rows, m - columns
vector<vector<double>> transposeMatrix(const vector<vector<double>> A) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> AT(m, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

// Helper function to perform matrix multiplication, complexity O(n*m*p), n - rows, m - columns, p - columns
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

// Helper function to perform matrix subtraction, complexity O(n*m), n - rows, m - columns
vector<vector<double>> matrixSubtract(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> C(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
};

// Helper function to calculate the norm of a vector, complexity O(n)
double norm(const vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

// Helper function to perform scalar multiplication, complexity O(n)
vector<double> scalarMultiply(const vector<double>& x, double alpha) {
    vector<double> result(x.size(), 0.0);
    for (int i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i];
    }
    return result;
}

// Helper function to create an identity matrix, complexity O(n)
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
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            R[i][j] = 0.0;
        }
    }
    return make_pair(Q, R);
}

vector<double> calculateEigenvalues(vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> ATemp = matrix;
    for (int i = 0; i < 1000; ++i) {
        pair<vector<vector<double>>, vector<vector<double>>> QR = householderTransformation(ATemp);
        ATemp = matrixMultiply(QR.second, QR.first);    
    }
    vector<double> eigenvalues(n, 0.0);
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = ATemp[i][i];
    }
    return eigenvalues;
}

std::vector<double> backSubstitution(const std::vector<std::vector<double>>& matrix, const std::vector<double>& b) {
    int n = matrix.size();
    std::vector<double> x(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += matrix[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / matrix[i][i];
    }

    return x;
}

vector<double> solveLinearEquations(vector<vector<double>> A, vector<double> b) {
    int n = A.size();

    
    for (int i = 0; i < n; i++) {
        // Make the diagonal element non-zero
        if (A[i][i] == 0) {
            cout << "Error: Diagonal element is zero, cannot proceed." << endl;
            exit(1);
        }

        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < n; k++) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    A[n-1][n-1] = 1.0;
    b[n-1] = 1.0;

    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    return x;
}

vector<vector<double>> calculateEigenvectors(vector<vector<double>>& matrix, vector<double>& eigenvalues) {
    int n = matrix.size();
    vector<vector<double>> eigenvectors(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        vector<vector<double>> A = matrix;
        for (int j = 0; j < n; ++j) {
            A[j][j] -= eigenvalues[i];
        }
        vector<double> b(n, 0.0);
        vector<double> x = solveLinearEquations(transposeMatrix(A), b);
        for (int j = 0; j < n; ++j) {
            eigenvectors[j][i] = x[j];
        }
    }
    return eigenvectors;
}

pair<vector<double>, vector<vector<double>>> sortEigenvalues(vector<double> eigenvalues, vector<vector<double>> eigenvectors){
    vector<pair<double, vector<double>>> eigenPairs;
    for (int i = 0; i < eigenvalues.size(); i++) {
        eigenPairs.push_back(make_pair(eigenvalues[i], eigenvectors[i]));
    }
    sort(eigenPairs.begin(), eigenPairs.end(), greater<pair<double, vector<double>>>());
    vector<double> sortedEigenvalues;
    vector<vector<double>> sortedEigenvectors;
    for (int i = 0; i < eigenvalues.size(); i++) {
        sortedEigenvalues.push_back(eigenPairs[i].first);
        sortedEigenvectors.push_back(eigenPairs[i].second);
    }
    return make_pair(sortedEigenvalues, sortedEigenvectors);
}

vector<double> calculateCumulativeSum(vector<double>& vec) {
    vector<double> cumulativeSum(vec.size(), 0.0);

    cumulativeSum[0] = vec[0];
    for (int i = 1; i < vec.size(); i++) {
        cumulativeSum[i] = cumulativeSum[i - 1] + vec[i];
    }
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    for (int i = 0; i < vec.size(); i++) {
        cumulativeSum[i] /= sum;
    }
    return cumulativeSum;
}

int findThresholdIndex(vector<double>& cumulativeSum, double threshold) {
    for (int i = 0; i < cumulativeSum.size(); ++i) {
        if (cumulativeSum[i] >= threshold) {
            return i;
        }
    }
    return cumulativeSum.size() - 1;
}

vector<vector<double>> pca(vector<vector<double>>& data) {
    vector<vector<double>> standardizedData;
    for (int i = 0; i < data[0].size(); i++) {
        vector<double> column;
        for (int j = 0; j < data.size(); j++) {
            column.push_back(data[j][i]);
        }
        standardizedData.push_back(standardize(column));
    }
    vector<vector<double>> covMatrix = covarianceMatrix(standardizedData);
    vector<double> eigenvalues = calculateEigenvalues(covMatrix);
    vector<vector<double>> eigenvectors = calculateEigenvectors(covMatrix, eigenvalues);
    pair<vector<double>, vector<vector<double>>> sortedEigenvaluesVectors = sortEigenvalues(eigenvalues, eigenvectors);
    vector<double> sortedEigenvalues = sortedEigenvaluesVectors.first;
    vector<vector<double>> sortedEigenvectors = sortedEigenvaluesVectors.second;
    vector<double> explainedVariance = calculateCumulativeSum(eigenvalues);
    int thresholdIndex = findThresholdIndex(explainedVariance, 0.8);
    vector<vector<double>> usefulComponents;
    for (int i = 0; i < thresholdIndex + 1; i++) {
        usefulComponents.push_back(eigenvectors[i]);
    }
    return matrixMultiply(usefulComponents, standardizedData);
}



int main(){
    // create array of strings for file names
    string fileNames[] = {"data/data_1.csv", "data/data_2.csv", "data/data_3.csv", "data/data_4.csv", 
            "data/data_5.csv", "data/data_6.csv", "data/data_7.csv", "data/data.csv"};
    for (int i = 0; i < 8; i++) {
        vector<vector<double>> data = readData(fileNames[i]);
        auto start_time = chrono::high_resolution_clock::now();
        vector<vector<double>> pcaData = pca(data);
        auto end_time = chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        cout << "Time taken: " << time/chrono::milliseconds(1) << " ms" << endl;
    }

    return 0;
}
