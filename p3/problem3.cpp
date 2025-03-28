#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <x86intrin.h>
#include <eigen3/Eigen/Dense>
#include <omp.h>


double* generate_matrix_arr(const int size, const bool align = false) {
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

    double* matrix;
    if (align) {
        matrix = new(std::align_val_t(32)) double[size * size];
    } else {
        matrix = new double[size * size];
    }
     
    for (int i = 0; i < size*size; i++) {
        matrix[i] = uniform_distribution(generator);
    }
    return matrix;
}

double* generate_empty_arr(const int size, const bool align = false) {
    if (align) {
        return new(std::align_val_t(32)) double[size * size];
    } else {
        return new double[size * size];
    }
}

void print_matrix_arr(const double* matrix, const int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            std::cout << matrix[row * size + col] << " ";
        }
        std::cout << '\n';
    }
}

void matmul_inner(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {

                double* b_pack = new(std::align_val_t(32)) double[block_size * block_size];
                for (int k = 0; k < block_size; k++) {
                    for (int j = 0; j < block_size; j++) {
                        b_pack[k * block_size + j] = matrix_b[(k_block + k) * size + j_block + j];
                    }
                }

                #pragma omp parallel for
                for (int i = 0; i < block_size; i++) {
                    for (int k = 0; k < block_size; k++) {
                        __m256d a_vector = _mm256_broadcast_sd(&matrix_a[(i_block + i) * size + k_block + k]);
                        for (int j = 0; j < block_size; j += 4) {
                            __m256d b_vector = _mm256_load_pd(&b_pack[k * block_size + j]);
                            __m256d result_vector = _mm256_load_pd(&result_matrix[(i_block + i) * size + j_block + j]);
                            result_vector = _mm256_add_pd(result_vector, _mm256_mul_pd(a_vector, b_vector));
                            _mm256_store_pd(&result_matrix[(i_block + i) * size + j_block + j], result_vector);
                        }
                    }
                }

                delete[] b_pack;
            }
        } 
    }  
}

void matmul_outer(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    #pragma omp parallel for
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {

                double* b_pack = new(std::align_val_t(32)) double[block_size * block_size];
                for (int k = 0; k < block_size; k++) {
                    for (int j = 0; j < block_size; j++) {
                        b_pack[k * block_size + j] = matrix_b[(k_block + k) * size + j_block + j];
                    }
                }

                for (int i = 0; i < block_size; i++) {
                    for (int k = 0; k < block_size; k++) {
                        __m256d a_vector = _mm256_broadcast_sd(&matrix_a[(i_block + i) * size + k_block + k]);
                        for (int j = 0; j < block_size; j += 4) {
                            __m256d b_vector = _mm256_load_pd(&b_pack[k * block_size + j]);
                            __m256d result_vector = _mm256_load_pd(&result_matrix[(i_block + i) * size + j_block + j]);
                            result_vector = _mm256_add_pd(result_vector, _mm256_mul_pd(a_vector, b_vector));
                            _mm256_store_pd(&result_matrix[(i_block + i) * size + j_block + j], result_vector);
                        }
                    }
                }

                delete[] b_pack;
            }
        } 
    }  
}

void matmul_outer_schedule(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    #pragma omp parallel for schedule(static, 1)
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {

                double* b_pack = new(std::align_val_t(32)) double[block_size * block_size];
                for (int k = 0; k < block_size; k++) {
                    for (int j = 0; j < block_size; j++) {
                        b_pack[k * block_size + j] = matrix_b[(k_block + k) * size + j_block + j];
                    }
                }

                for (int i = 0; i < block_size; i++) {
                    for (int k = 0; k < block_size; k++) {
                        __m256d a_vector = _mm256_broadcast_sd(&matrix_a[(i_block + i) * size + k_block + k]);
                        for (int j = 0; j < block_size; j += 4) {
                            __m256d b_vector = _mm256_load_pd(&b_pack[k * block_size + j]);
                            __m256d result_vector = _mm256_load_pd(&result_matrix[(i_block + i) * size + j_block + j]);
                            result_vector = _mm256_add_pd(result_vector, _mm256_mul_pd(a_vector, b_vector));
                            _mm256_store_pd(&result_matrix[(i_block + i) * size + j_block + j], result_vector);
                        }
                    }
                }

                delete[] b_pack;
            }
        } 
    }  
}


void test(double* matrix_a, double* matrix_b, double* result_matrix, const int N) {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_result(result_matrix, N, N);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_matrix_a(matrix_a, N, N);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_matrix_b(matrix_b, N, N);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_computed = eigen_matrix_a * eigen_matrix_b;

    double error = (eigen_computed - eigen_result).norm();
    std::cout << error << '\n';
}

int main() {
    omp_set_num_threads(16);

    const int N = 2048;
    const int BLOCK_SIZE = 128; 
    
    auto matrix_a = generate_matrix_arr(N, true);
    auto matrix_b = generate_matrix_arr(N, true);

    auto result_matrix = generate_empty_arr(N, true);

    auto start_time = std::chrono::steady_clock::now();
    matmul_outer_schedule(matrix_a, matrix_b, result_matrix, N, BLOCK_SIZE);
    auto end_time = std::chrono::steady_clock::now();
    int execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << execution_time << '\n';

    test(matrix_a, matrix_b, result_matrix, N);
    
    delete[] matrix_a;
    delete[] matrix_b;
    delete[] result_matrix;
    
    return 0;
}