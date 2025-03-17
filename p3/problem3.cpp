#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <omp.h>

double* generate_matrix_arr(const int size) {
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

    double* matrix = new double[size * size];
    for (int i = 0; i < size*size; i++) {
        matrix[i] = uniform_distribution(generator);
    }
    return matrix;
}

void print_matrix_arr(const double* matrix, const int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            std::cout << matrix[row * size + col] << " ";
        }
        std::cout << '\n';
    }
}



void multiply_ikj_arr(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size) {     
    #pragma omp parallel for 
    for (int i = 0; i < size; i++) {  
        for (int k = 0; k < size; k++) {
            const double a_value = matrix_a[i * size + k];
            for (int j = 0; j < size; j++) {
                result_matrix[i * size + j] += a_value * matrix_b[k * size + j];
            }
        }
    }
}

void multiply_ikj_arr_blocking(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    #pragma omp parallel for
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {
                for (int i = 0; i < block_size; i++) {
                    for (int k = 0; k < block_size; k++) {
                        const double a_value = matrix_a[(i_block + i) * size + (k_block + k)];
                        for (int j = 0; j < block_size; j++) {
                            result_matrix[(i_block + i) * size + (j_block + j)] += a_value * matrix_b[(k_block + k) * size + (j_block + j)];
                        }
                    }
                }
            } 
        } 
    }  
}

void multiply_ikj_arr_packing_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    #pragma omp parallel for
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {

            double* a_pack = new double[block_size * block_size];
            for (int i = 0; i < block_size; i++) {
                for (int k = 0; k < block_size; k++) {
                    a_pack[i * block_size + k] = matrix_a[(i_block + i) * size + k_block + k];
                }
            }

            for (int j_block = 0; j_block < size; j_block += block_size) {

                double* b_pack = new double[block_size * block_size];
                for (int k = 0; k < block_size; k++) {
                    for (int j = 0; j < block_size; j++) {
                        b_pack[k * block_size + j] = matrix_b[(k_block + k) * size + j_block + j];
                    }
                }

                for (int i = 0; i < block_size; i++) {
                    for (int k = 0; k < block_size; k++) {
                        const double a_value = a_pack[i * block_size + k];
                        for (int j = 0; j < block_size; j++) {
                            result_matrix[(i_block + i) * size + (j_block + j)] += a_value * b_pack[k * block_size + j];
                        }
                    }
                }

                delete[] b_pack;
            }
            
            delete[] a_pack;
        } 
    }  
}

void multiply_ikj_arr_blocking_non_square_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                                 const int size, const int i_block_size, const int j_block_size, const int k_block_size) {
    #pragma omp parallel for
    for (int i_block = 0; i_block < size; i_block += i_block_size) {
        for (int k_block = 0; k_block < size; k_block += k_block_size) {
            for (int j_block = 0; j_block < size; j_block += j_block_size) {
                for (int i = 0; i < i_block_size; i++) {
                    for (int k = 0; k < k_block_size; k++) {
                        const double a_value = matrix_a[(i_block + i) * size + (k_block + k)];
                        for (int j = 0; j < j_block_size; j++) {
                            result_matrix[(i_block + i) * size + (j_block + j)] += a_value * matrix_b[(k_block + k) * size + (j_block + j)];
                        }
                    }
                }
            } 
        } 
    }  
}

void multiply_ikj_arr_packing_non_square_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                          const int size, const int i_block_size, const int j_block_size, const int k_block_size) {
    #pragma omp parallel for
    for (int i_block = 0; i_block < size; i_block += i_block_size) {
        for (int k_block = 0; k_block < size; k_block += k_block_size) {

            double* a_pack = new double[i_block_size * k_block_size];
            for (int i = 0; i < i_block_size; i++) {
                for (int k = 0; k < k_block_size; k++) {
                    a_pack[i * k_block_size + k] = matrix_a[(i_block + i) * size + k_block + k];
                }
            }

            for (int j_block = 0; j_block < size; j_block += j_block_size) {

                double* b_pack = new double[k_block_size * j_block_size];
                for (int k = 0; k < k_block_size; k++) {
                    for (int j = 0; j < j_block_size; j++) {
                        b_pack[k * j_block_size + j] = matrix_b[(k_block + k) * size + j_block + j];
                    }
                }

                for (int i = 0; i < i_block_size; i++) {
                    for (int k = 0; k < k_block_size; k++) {
                        const double a_value = a_pack[i * k_block_size + k];
                        for (int j = 0; j < j_block_size; j++) {
                            result_matrix[(i_block + i) * size + (j_block + j)] += a_value * b_pack[k * j_block_size + j];
                        }
                    }
                }

                delete[] b_pack;
            }
            
            delete[] a_pack;
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
    const int BLOCK_SIZE = 128;

    const int I_BLOCK_SIZE = 8;
    const int J_BLOCK_SIZE = 2048;
    const int K_BLOCK_SIZE = 2;
    
    
    double* matrix_a = generate_matrix_arr(N);
    double* matrix_b = generate_matrix_arr(N);

    double* result_matrix = new double[N * N]();

    auto start_time = std::chrono::steady_clock::now();
    multiply_ikj_arr_packing_simple(matrix_a, matrix_b, result_matrix, N, BLOCK_SIZE);
    auto end_time = std::chrono::steady_clock::now();
    int execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << execution_time << '\n';

    test(matrix_a, matrix_b, result_matrix, N);
    
    delete[] matrix_a;
    delete[] matrix_b;
    delete[] result_matrix;
    
    
    
    /*
    int best_time = 100000;
    int best_i = 0;
    int best_j = 0;
    int best_k = 0;

    for (int i = 32; i <= 512; i *= 2) {
        for (int j = 32; j <= 512; j *= 2) {
            for (int k = 32; k <= 512; k *= 2) {
                double* matrix_a = generate_matrix_arr(2048);
                double* matrix_b = generate_matrix_arr(2048);

                double* result_matrix = new double[2048*2048]();

                auto start_time = std::chrono::steady_clock::now();
                multiply_ikj_arr_packing_non_square_simple(matrix_a, matrix_b, result_matrix, 2048, i, j, k);
                auto end_time = std::chrono::steady_clock::now();
                int execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

                std::cout << "i=" << i << " j=" << j << " k=" << k << ": " << execution_time << "\n";

                if (execution_time < best_time) {
                    best_time = execution_time;
                    best_i = i;
                    best_j = j;
                    best_k = k;
                }

                delete[] matrix_a;
                delete[] matrix_b;
                delete[] result_matrix;
            }
        }
    }
        
    std::cout << "BEST: i=" << best_i << " j=" << best_j << " k=" << best_k << ": " << best_time << "\n";
    */
    
    return 0;
}