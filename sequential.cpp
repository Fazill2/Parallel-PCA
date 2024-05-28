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

void saveData(vector<vector<double>> data, string filename) {
    ofstream file;
    file.open(filename);
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            file << data[j][i] << ",";
        }
        file << endl;
    }
    file.close();
}

double** vectorMatrixToDouble(vector<vector<double>>& data) {
    int rows = data.size();
    int columns = data[0].size();
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new double[columns];
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = data[i][j];
        }
    }
    return matrix;
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
    vector<double> standardized(vec.size());
    for (int i = 0; i < vec.size(); i++) {
        standardized[i] = (vec[i] - m) / sd;
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
vector<vector<double>> transposeMatrix(vector<vector<double>> A) {
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
vector<vector<double>> matrixMultiply(vector<vector<double>>& A, vector<vector<double>>& B) {
    int n = A.size();
    int m = B.size();
    int p = B[0].size();
    double** matrixA = vectorMatrixToDouble(A);
    double** matrixB = vectorMatrixToDouble(B);
    double** matrixC = new double*[n];
    for (int i = 0; i < n; i++) {
        matrixC[i] = new double[p]();
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < m; ++k) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    vector<vector<double>> C(n, vector<double>(p, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = matrixC[i][j];
        }
    }
    

    return C;
}

// Helper function to perform matrix subtraction, complexity O(n*m), n - rows, m - columns
vector<vector<double>> matrixSubtract(vector<vector<double>>& A, vector<vector<double>>& B) {
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
double norm(vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

// Helper function to perform scalar multiplication, complexity O(n)
vector<double> scalarMultiply(vector<double>& x, double alpha) {
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
pair<vector<vector<double>>, vector<vector<double>>> householderTransformation(vector<vector<double>>& A) {
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
    return make_pair(Q, R);
}


vector<double> calculateEigenvalues(vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> ATemp = matrix;
    
    bool converged = false;
    for (;;) {
        pair<vector<vector<double>>, vector<vector<double>>> QR = householderTransformation(ATemp);
        vector<vector<double>> new_ATemp = matrixMultiply(QR.second, QR.first);
        for (int i = 0; i < n; ++i) {
            bool flag = false;
            for (int j = 0; j < n; ++j) {
                if (abs(new_ATemp[i][j] - ATemp[i][j]) > 1e-6) {
                    flag = true;
                    break;
                }
                if (i == n - 1 && j == n - 1) {
                    converged = true;
                }
            }
            if (flag) {
                break;
            }
        }
        if (converged){
            break;
        }
        ATemp = new_ATemp;
    }
    vector<double> eigenvalues(n, 0.0);
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = ATemp[i][i];
    }
    return eigenvalues;
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
            eigenvectors[i][j] = x[j];
        }
    }
    return eigenvectors;
}

pair<vector<double>, vector<vector<double>>> sortEigenvalues(vector<double> eigenvalues, vector<vector<double>> eigenvectors){
    vector<pair<double, vector<double>>> eigenPairs(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); i++) {
        eigenPairs[i] = make_pair(eigenvalues[i], eigenvectors[i]);
    }
    sort(eigenPairs.begin(), eigenPairs.end(), greater<pair<double, vector<double>>>());
    vector<double> sortedEigenvalues = eigenvalues;
    vector<vector<double>> sortedEigenvectors = eigenvectors;
    for (int i = 0; i < eigenvalues.size(); i++) {
        sortedEigenvalues[i] = eigenPairs[i].first;
        sortedEigenvectors[i] = eigenPairs[i].second;
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
    vector<vector<double>> standardizedData(data[0].size());
    for (int i = 0; i < data[0].size(); i++) {
        vector<double> column(data.size());
        for (int j = 0; j < data.size(); j++) {
            column[j] = data[j][i];
        }
        standardizedData[i] = standardize(column);
    }
    vector<vector<double>> covMatrix = covarianceMatrix(standardizedData);
    vector<double> eigenvalues = calculateEigenvalues(covMatrix);

    vector<vector<double>> eigenvectors = calculateEigenvectors(covMatrix, eigenvalues);
    pair<vector<double>, vector<vector<double>>> sortedEigenvaluesVectors = sortEigenvalues(eigenvalues, eigenvectors);
    vector<double> sortedEigenvalues = sortedEigenvaluesVectors.first;
    vector<vector<double>> sortedEigenvectors = sortedEigenvaluesVectors.second;
    vector<double> explainedVariance = calculateCumulativeSum(eigenvalues);
    int thresholdIndex = findThresholdIndex(explainedVariance, 0.8);
    vector<vector<double>> usefulComponents(thresholdIndex + 1);
    for (int i = 0; i < thresholdIndex + 1; i++) {
        // normalize the eigenvectors
        double norm = sqrt(inner_product(sortedEigenvectors[i].begin(), sortedEigenvectors[i].end(), sortedEigenvectors[i].begin(), 0.0));
        for (int j = 0; j < sortedEigenvectors[i].size(); j++) {
            sortedEigenvectors[i][j] /= norm;
        }
        usefulComponents[i] = sortedEigenvectors[i];
    }
    return matrixMultiply(usefulComponents, standardizedData);
}



int main(){
    // create array of strings for file names
    string fileNames[] = {"randomData/data_0.csv", "randomData/data_1.csv", "randomData/data_2.csv", "randomData/data_3.csv", 
            "randomData/data_4.csv", "randomData/data_5.csv", "randomData/data_6.csv", "randomData/data_7.csv"};
    // string fileNames[] = {"data/data.csv"};
    
    for (int i = 0; i < 8; i++) {
        vector<vector<double>> data = readData(fileNames[i]);
        auto start_time = chrono::high_resolution_clock::now();
        vector<vector<double>> pcaData = pca(data);
        auto end_time = chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        cout << "Time taken: " << time/chrono::milliseconds(1) << " ms" << endl;
        saveData(pcaData, "./output/output_" + to_string(i + 1) + ".csv");
    }

    return 0;
}
