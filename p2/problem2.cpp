#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <x86intrin.h>
#include <eigen3/Eigen/Dense>
#include <new>

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

std::vector<std::vector<double>> generate_empty(const int size) {
    std::vector<std::vector<double>> result_matrix(size, std::vector<double>(size, 0));
    return result_matrix;
}

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
        return new(std::align_val_t(32)) double[size * size]{};
    } else {
        return new double[size * size]{};
    }
}

void print_matrix(const std::vector<std::vector<double>>& matrix) {
    for (auto& row: matrix) {
        for (double value: row) {
            std::cout << value << " ";
        }
        std::cout << '\n';
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



void matmul_ijk(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
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

void matmul_jik(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result_matrix[i][j] = sum;
        }
    }
}

void matmul_ikj(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            const double a_value = matrix_a[i][k];
            for (int j = 0; j < size; j++) {
                result_matrix[i][j] += a_value * matrix_b[k][j];
            }
        }
    }
}

void matmul_kij(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            const double a_value = matrix_a[i][k];
            for (int j = 0; j < size; j++) {
                result_matrix[i][j] += a_value * matrix_b[k][j];
            }
        }
    }
}

void matmul_jki(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int j = 0; j < size; j++) {
        for (int k = 0; k < size; k++) {
            const double b_value = matrix_b[k][j];
            for (int i = 0; i < size; i++) {
                result_matrix[i][j] += b_value * matrix_a[i][k];
            }
        }
    }
}

void matmul_kji(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
            const double b_value = matrix_b[k][j];
            for (int i = 0; i < size; i++) {
                result_matrix[i][j] += b_value * matrix_a[i][k];
            }
        }
    }
}

void matmul_ikj_arr(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size) {       
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            const double a_value = matrix_a[i * size + k];
            for (int j = 0; j < size; j++) {
                result_matrix[i * size + j] += a_value * matrix_b[k * size + j];
            }
        }
    }
}

void matmul_ikj_arr_blocking(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {
                for (int i = i_block; i < i_block + block_size; i++) {
                    for (int k = k_block; k < k_block + block_size; k++) {
                        const double a_value = matrix_a[i * size + k];
                        for (int j = j_block; j < j_block + block_size; j++) {
                            result_matrix[i * size + j] += a_value * matrix_b[k * size + j];
                        }
                    }
                }
            } 
        } 
    }  
}

void matmul_ikj_arr_blocking_vector(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {
                for (int i = i_block; i < i_block + block_size; i++) {
                    for (int k = k_block; k < k_block + block_size; k++) {
                        __m256d a_vector = _mm256_broadcast_sd(&matrix_a[i * size + k]);
                        for (int j = j_block; j < j_block + block_size; j += 4) {
                            __m256d b_vector = _mm256_loadu_pd(&matrix_b[k * size + j]);
                            __m256d result_vector = _mm256_loadu_pd(&result_matrix[i * size + j]);
                            result_vector = _mm256_add_pd(result_vector, _mm256_mul_pd(a_vector, b_vector));
                            _mm256_storeu_pd(&result_matrix[i * size + j], result_vector);
                        }
                    }
                }
            } 
        } 
    }  
}

void matmul_ikj_arr_blocking_vector_aligned(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {
                for (int i = i_block; i < i_block + block_size; i++) {
                    for (int k = k_block; k < k_block + block_size; k++) {
                        __m256d a_vector = _mm256_broadcast_sd(&matrix_a[i * size + k]);
                        for (int j = j_block; j < j_block + block_size; j += 4) {
                            __m256d b_vector = _mm256_load_pd(&matrix_b[k * size + j]);
                            __m256d result_vector = _mm256_load_pd(&result_matrix[i * size + j]);
                            result_vector = _mm256_add_pd(result_vector, _mm256_mul_pd(a_vector, b_vector));
                            _mm256_store_pd(&result_matrix[i * size + j], result_vector);
                        }
                    }
                }
            } 
        } 
    }  
}

void matmul_ikj_arr_blocking_vector_aligned_packing(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
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
    const int N = 2048;
    const int BLOCK_SIZE = 32;
    
    auto matrix_a = generate_matrix(N);
    auto matrix_b = generate_matrix(N);

    auto result_matrix = generate_empty(N);

    auto start_time = std::chrono::steady_clock::now();
    matmul_ijk(matrix_a, matrix_b, result_matrix, N);
    auto end_time = std::chrono::steady_clock::now();
    int execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << execution_time << '\n';
    
    //test(matrix_a, matrix_b, result_matrix, N);
    
    /*
    delete[] matrix_a;
    delete[] matrix_b;
    delete[] result_matrix;
    */
    
    return 0;
}