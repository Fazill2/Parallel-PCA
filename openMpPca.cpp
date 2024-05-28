#include <omp.h>
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

void generateRandomData(int rows, int columns, string filename) {   
    ofstream file;
    vector<vector<double>> data(columns, vector<double>(rows));
    
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            data[i][j] = (double)rand() / RAND_MAX;
        }
    }
    // for random number of random columns generate random linear equation ax+b where x is another random column
    int randomColumns = rand() % columns;
    vector<double> a(randomColumns);
    vector<double> b(randomColumns);
    for (int i = 0; i < randomColumns; i++) {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
    }
    // pca requires data to be somewhat dependent
    for (int i = 0; i < randomColumns; i++) {
        int randColumn1 = rand() % columns;
        int randColumn2 = rand() % columns;
        for (int j = 0; j < rows; j++) {
            data[randColumn1][j] = a[i] * data[randColumn2][j] + b[i];
        }
    }

    file.open(filename);
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            file << data[j][i];
            if (j != data.size() - 1) {
                file << ",";
            }
        }
        if (i != data[0].size() - 1) {
            file << endl;
        }
    }
    file.close();
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
    #pragma omp parallel for collapse(2)
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
    #pragma omp parallel for
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
    vector<vector<double>> C(n, vector<double>(p, 0.0));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0;
            for (int k = 0; k < m; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    // double** matrixA = vectorMatrixToDouble(A);
    // double** matrixB = vectorMatrixToDouble(B);
    // double** matrixC = new double*[n];
    // for (int i = 0; i < n; i++) {
    //     matrixC[i] = new double[p]();
    // }
    
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < p; ++j) {
    //         double sum = 0;
    //         for (int k = 0; k < m; ++k) {
    //             sum += matrixA[i][k] * matrixB[k][j];
    //         }
    //         matrixC[i][j] = sum;
    //     }
    // }
    // vector<vector<double>> C(n, vector<double>(p, 0.0));
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < p; j++) {
    //         C[i][j] = matrixC[i][j];
    //     }
    // }
    // for (int i = 0; i < n; i++) {
    //     delete[] matrixA[i];
    // }
    // delete[] matrixA;
    // for (int i = 0; i < m; i++) {
    //     delete[] matrixB[i];
    // }
    // delete[] matrixB;
    // for (int i = 0; i < n; i++) {
    //     delete[] matrixC[i];
    // }
    // delete[] matrixC;
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

pair<vector<vector<double>>, vector<vector<double>>> householderTransformation(vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> Q = identityMatrix(n);
    vector<vector<double>> R = A;

    for (int i = 0; i < n - 1; ++i) {
        vector<double> x(n - i, 0.0);
        for (int j = i; j < n; ++j) {
            x[j - i] = R[j][i];
        }

        double alpha = (x[0] >= 0) ? -norm(x) : norm(x);

        vector<double> e(n - i, 0.0);
        e[0] = 1.0;

        vector<double> v(n - i);
        for (int j = 0; j < n - i; ++j) {
            v[j] = x[j] + alpha * e[j];
        }

        double vNorm = norm(v);
        if (vNorm != 0) {
            v = scalarMultiply(v, 1.0 / vNorm);
        }

        for (int j = i; j < n; ++j) {
            double dotProduct = 0.0;
            for (int k = i; k < n; ++k) {
                dotProduct += v[k - i] * R[k][j];
            }
            for (int k = i; k < n; ++k) {
                R[k][j] -= 2.0 * v[k - i] * dotProduct;
            }
        }

        for (int j = 0; j < n; ++j) {
            double dotProduct = 0.0;
            for (int k = i; k < n; ++k) {
                dotProduct += v[k - i] * Q[j][k];
            }
            for (int k = i; k < n; ++k) {
                Q[j][k] -= 2.0 * v[k - i] * dotProduct;
            }
        }
    }
    return make_pair(Q, R);
}


vector<double> calculateEigenvalues(vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> ATemp = matrix;
    
    const double epsilon = 1e-4;
    bool converged = false;
    
    while (!converged) {
        pair<vector<vector<double>>, vector<vector<double>>> QR = householderTransformation(ATemp);
        vector<vector<double>> new_ATemp = matrixMultiply(QR.second, QR.first);
        
        converged = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (abs(new_ATemp[i][j] - ATemp[i][j]) > epsilon) {
                    converged = false;
                    break;
                }
            }
            if (!converged) {
                break;
            }
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
    #pragma omp parallel for
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
    string fileNames[] = {"randomData/data_0.csv", "randomData/data_1.csv", "randomData/data_2.csv", "randomData/data_3.csv", 
        "randomData/data_4.csv", "randomData/data_5.csv", "randomData/data_6.csv", "randomData/data_7.csv", 
        "randomData/data_8.csv", "randomData/data_9.csv", "randomData/data_10.csv", "randomData/data_11.csv"};
    int rows = 1000;
    int columns[] = {10, 20, 30 ,40, 50, 60, 70, 80, 90, 100, 110, 120};
    float times[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // for (int i = 0; i < 12; i++) {
    //     generateRandomData(rows, columns[i], fileNames[i]);
    // }
    for (int i = 0; i < 12; i++) {
        vector<vector<double>> data = readData(fileNames[i]);
        auto start_time = chrono::high_resolution_clock::now();
        vector<vector<double>> pcaData = pca(data);
        auto end_time = chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        int timeTaken = chrono::duration_cast<chrono::milliseconds>(time).count();
        cout << "Time taken: " << time/chrono::milliseconds(1) << " ms" << endl;
        times[i] =  (float)timeTaken / 1000.0;
        saveData(pcaData, "./output/output_" + to_string(i + 1) + ".csv");
    }
    // save times to file for plotting
    ofstream file;
    file.open("ompTimes.txt");
    file << "[";
    for (int i = 0; i < 12; i++) {
        file << times[i];
        if (i != 11) {
            file << ",";
        }
    }
    file << "]";
    return 0;
}
