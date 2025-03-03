#include <iostream>
#include <vector>
#include <random>
#include <chrono>

std::vector<std::vector<double>> generate_matrix(const int size) {
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

    std::vector<std::vector<double>> matrix(size, std::vector<double>(size));
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            matrix[row][col] = uniform_distribution(generator);
        }
    }
    return matrix;
}

void print_matrix(const std::vector<std::vector<double>>& matrix) {
    for (auto& row: matrix) {
        for (double value: row) {
            std::cout << value << " ";
        }
        std::cout << '\n';
    }
}

void multiply(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result_matrix[i][j] = sum;
        }
    }
}

int main() {
    const int N = 1024;

    auto matrix_a = generate_matrix(N);
    auto matrix_b = generate_matrix(N);
    std::vector<std::vector<double>> result_matrix(N, std::vector<double>(N)); 

    auto start_time = std::chrono::steady_clock::now();
    multiply(matrix_a, matrix_b, result_matrix, N);
    auto end_time = std::chrono::steady_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << execution_time << '\n';
    return 0;
}