#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

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
#include <filesystem>
#include <string>
#include <algorithm>
#include <regex>
#include <experimental/filesystem>
using namespace std;
namespace fs = experimental::filesystem;

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

double mean(vector<double> vec) {
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

double standardDeviation(vector<double> vec) {
    double m = mean(vec);
    double sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += pow(vec[i] - m, 2);
    }
    return sqrt(sum / vec.size());
}

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
   
    pair<vector<double>, vector<vector<double>>> sortedEigenvaluesVectors = sortEigenvalues(eigenvalues, eigenvectors);
    vector<double> sortedEigenvalues = sortedEigenvaluesVectors.first;
    vector<vector<double>> sortedEigenvectors = sortedEigenvaluesVectors.second;
    vector<double> explainedVariance = calculateCumulativeSum(sortedEigenvalues);
    int thresholdIndex = findThresholdIndex(explainedVariance, 0.8);
    vector<vector<double>> usefulComponents(thresholdIndex + 1);

    for (int i = 0; i < thresholdIndex + 1; i++) {
        double norm = sqrt(inner_product(sortedEigenvectors[i].begin(), sortedEigenvectors[i].end(), sortedEigenvectors[i].begin(), 0.0));
        for (int j = 0; j < sortedEigenvectors[i].size(); j++) {
            sortedEigenvectors[i][j] /= norm;
        }
        usefulComponents[i] = sortedEigenvectors[i];
    }
    return matrixMultiply(usefulComponents, standardizedData);
}


int extractNumber(const std::string& filename) {
    std::regex re(R"(_(\d+)\.csv$)");
    std::smatch match;
    if (std::regex_search(filename, match, re) && match.size() > 1) {
        return std::stoi(match.str(1));
    }
    return -1;
}

int main(){
    string path = ".\\randomData";

    vector<fs::directory_entry> files;
    for (const auto& entry : fs::directory_iterator(path)) {
        files.push_back(entry);
    }

    // Sort files based on the numerical part of the filenames
    sort(files.begin(), files.end(), [](const fs::directory_entry& a, const fs::directory_entry& b) {
        return extractNumber(a.path().filename().string()) < extractNumber(b.path().filename().string());
    });
    int i = 0;
    vector<float> times;
    for (const auto& file : files) {
        vector<vector<double>> data = readData(file.path().string());
        auto start_time = chrono::high_resolution_clock::now();
        vector<vector<double>> pcaData = pca(data);
        auto end_time = chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        int timeTaken = chrono::duration_cast<chrono::milliseconds>(time).count();
        cout << "Time taken: " << time/chrono::milliseconds(1) << " ms" << endl;
        times.push_back(timeTaken);
        saveData(pcaData, "./output/output_" + to_string(++i) + ".csv");
    }
    
    ofstream file;
    file.open("cudaTimes.txt");
    file << "[";
    for (int i = 0; i < times.size(); i++) {
        file << times[i];
        if (i != times.size() - 1) {
            file << ",";
        }
    }
    file << "]";
    return 0;
}
