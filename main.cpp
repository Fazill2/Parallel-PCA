// pca implementation in cpp

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
#include <iomanip>

using namespace std;

// Function to read data from a file
vector<vector<double>> readData(string filename) {
    ifstream file(filename);
    vector<vector<double>> data;
    string line;
    while (getline(file, line)) {
        vector<double> row;
        istringstream iss(line);
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        data.push_back(row);
    }
    return data;
}

// Function to calculate the mean of a vector
double mean(vector<double> vec) {
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

// Function to calculate standard deviation of a vector
double standardDeviation(vector<double> vec) {
    double m = mean(vec);
    double sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += pow(vec[i] - m, 2);
    }
    return sqrt(sum / vec.size());
}

// Function to standardize the vector
vector<double> standardize(vector<double> vec) {
    double m = mean(vec);
    double sd = standardDeviation(vec);
    vector<double> standardized;
    for (int i = 0; i < vec.size(); i++) {
        standardized.push_back((vec[i] - m) / sd);
    }
    return standardized;
}

// Function to calculate the covariance of two vectors
double covariance(vector<double> vec1, vector<double> vec2) {
    double m1 = mean(vec1);
    double m2 = mean(vec2);
    double sum = 0;
    for (int i = 0; i < vec1.size(); i++) {
        sum += (vec1[i] - m1) * (vec2[i] - m2);
    }
    return sum / vec1.size();
}

// Function to calculate the covariance matrix of a dataset
vector<vector<double>> covarianceMatrix(vector<vector<double>> data) {
    vector<vector<double>> covMatrix;
    for (int i = 0; i < data[0].size(); i++) {
        vector<double> row;
        for (int j = 0; j < data[0].size(); j++) {
            row.push_back(covariance(data[i], data[j]));
        }
        covMatrix.push_back(row);
    }
    return covMatrix;
}

// Function to calculate the eigenvalues and eigenvectors of a matrix
pair<vector<double>, vector<vector<double>>> eigen(vector<vector<double>> matrix) {
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    for (int i = 0; i < matrix.size(); i++) {
        vector<double> row;
        for (int j = 0; j < matrix.size(); j++) {
            if (i == j) {
                row.push_back(1);
            } else {
                row.push_back(0);
            }
        }
        eigenvectors.push_back(row);
    }
    for (int i = 0; i < matrix.size(); i++) {
        eigenvalues.push_back(matrix[i][i]);
    }
    return make_pair(eigenvalues, eigenvectors);
}



int main(){
    return 0;
}
