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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

using namespace std;

vector<double> flatten(const vector<vector<double>>& matrix) {
    vector<double> flattened;
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

pair<vector<double>, vector<vector<double>>> customEigenCuda(const vector<vector<double>>& covMatrix){
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    const int m = covMatrix.size();
    const int lda = m; 
    double* V = new double[lda*m]; // eigenvectors
    double* W = new double[m]; // eigenvalues
    double* A = new double[lda*m]; // covariance matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            A[i*lda + j] = covMatrix[i][j];
        }
    }

    double *d_A = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;

    int info_gpu = 0;

    cusolver_status = cusolverDnCreate(&cusolverH);

    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);

    cusolver_status = cusolverDnDsyevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo);

    cudaStat1 = cudaDeviceSynchronize();

    cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    vector<vector<double>> eigenVectors(m, vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            eigenVectors[i][j] = V[i*lda + j];
        }
    }
    vector<double> eigenValues(m);
    for (int i = 0; i < m; i++) {
        eigenValues[i] = W[i];
    }
    if (d_A ) cudaFree(d_A);
    if (d_W ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    delete[] V;
    delete[] W;
    delete[] A;

    return make_pair(eigenValues, eigenVectors);
}


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
    vector<vector<double>> C(n, vector<double>(p, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
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
    pair<vector<double>, vector<vector<double>> > eigen = customEigenCuda(covMatrix);
    vector<double> eigenvalues = eigen.first;
    vector<vector<double>> eigenvectors = eigen.second;

    // cout << "Eigenvalues: ";
    // for (int i = 0; i < eigenvalues.size(); i++) {
    //     cout << eigenvalues[i] << " ";
    // }
   
    pair<vector<double>, vector<vector<double>>> sortedEigenvaluesVectors = sortEigenvalues(eigenvalues, eigenvectors);
    vector<double> sortedEigenvalues = sortedEigenvaluesVectors.first;
    vector<vector<double>> sortedEigenvectors = sortedEigenvaluesVectors.second;
    vector<double> explainedVariance = calculateCumulativeSum(sortedEigenvalues);
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
    file.open("cudaTimes.txt");
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