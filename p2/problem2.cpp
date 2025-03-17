#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <eigen3/Eigen/Dense>

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




void multiply_ijk(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
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

void multiply_jik(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
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

void multiply_ikj(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            const double a_value = matrix_a[i][k];
            for (int j = 0; j < size; j++) {
                result_matrix[i][j] += a_value * matrix_b[k][j];
            }
        }
    }
}

void multiply_kij(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < size; i++) {
            const double a_value = matrix_a[i][k];
            for (int j = 0; j < size; j++) {
                result_matrix[i][j] += a_value * matrix_b[k][j];
            }
        }
    }
}

void multiply_jki(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int j = 0; j < size; j++) {
        for (int k = 0; k < size; k++) {
            const double b_value = matrix_b[k][j];
            for (int i = 0; i < size; i++) {
                result_matrix[i][j] += b_value * matrix_a[i][k];
            }
        }
    }
}

void multiply_kji(const std::vector<std::vector<double>>& matrix_a, const std::vector<std::vector<double>>& matrix_b, std::vector<std::vector<double>>& result_matrix, const int size) {       
    for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
            const double b_value = matrix_b[k][j];
            for (int i = 0; i < size; i++) {
                result_matrix[i][j] += b_value * matrix_a[i][k];
            }
        }
    }
}




void multiply_ikj_arr(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size) {       
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            const double a_value = matrix_a[i * size + k];
            for (int j = 0; j < size; j++) {
                result_matrix[i * size + j] += a_value * matrix_b[k * size + j];
            }
        }
    }
}

void multiply_ikj_arr2(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size) {       
    for (int i = 0; i < size; i++) {
        const int i_index = i * size;
        for (int k = 0; k < size; k++) {
            const int k_row_index = k * size;
            const double a_value = matrix_a[i_index + k];
            for (int j = 0; j < size; j++) {
                result_matrix[i_index + j] += a_value * matrix_b[k_row_index + j];
            }
        }
    }
}

void multiply_ikj_arr_flatten(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size) {       
    const int full_size = size*size;    
    for (int index = 0; index < full_size; index++) {
        const int i = (index / size) * size;
        const int k = (index % size) * size;
        const double a_value = matrix_a[index];
        for (int j = 0; j < size; j++) {
            result_matrix[i + j] += a_value * matrix_b[k + j];
        }
    }
}

void multiply_ikj_arr_blocking(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
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

void multiply_ikj_arr_blocking2(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {
            for (int j_block = 0; j_block < size; j_block += block_size) {
                const int full_block_size = block_size*block_size;    
                for (int index = 0; index < full_block_size; index++) {
                    const int i = (i_block + index / block_size) * size;
                    const int k = k_block + index % block_size;
                    const double a_value = matrix_a[i + k];
                    const int result_index = i + j_block;
                    const int b_index = k * size + j_block;
                    for (int j = 0; j < block_size; j++) {
                        result_matrix[result_index + j] += a_value * matrix_b[b_index + j];
                    }
                }
            } 
        } 
    }  
}

void multiply_ikj_arr_recursive(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                const int i_offset, const int j_offset, const int k_offset, 
                                const int size, const int full_size, const int block_size) {
    if (size <= block_size) {
        const int full_block_size = block_size*block_size;    
        for (int index = 0; index < full_block_size; index++) {
            const int i = (i_offset + index / block_size) * full_size;
            const int k = k_offset + index % block_size;
            const double a_value = matrix_a[i + k];
            const int result_index = i + j_offset;
            const int b_index = k * full_size + j_offset;
            for (int j = 0; j < block_size; j++) {
                result_matrix[result_index + j] += a_value * matrix_b[b_index + j];
            }
        } 
    } else {
        const int new_size = size / 2;
        const int new_i_offset = i_offset + new_size;
        const int new_j_offset = j_offset + new_size;
        const int new_k_offset = k_offset + new_size;

        // ikj
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, i_offset, j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, i_offset, new_j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, i_offset, j_offset, new_k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, i_offset, new_j_offset, new_k_offset, new_size, full_size, block_size);

        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, new_i_offset, j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, new_i_offset, new_j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, new_i_offset, j_offset, new_k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, new_i_offset, new_j_offset, new_k_offset, new_size, full_size, block_size);
    }
}

void multiply_ikj_arr_recursive(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    multiply_ikj_arr_recursive(matrix_a, matrix_b, result_matrix, 0, 0, 0, size, size, block_size);
}

void multiply_ikj_arr_packing(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {

            double* a_pack = new double[block_size * block_size];
            for (int i = 0; i < block_size; i++) {
                const int a_index = (i_block + i) * size + k_block;
                for (int k = 0; k < block_size; k++) {
                    a_pack[i * block_size + k] = matrix_a[a_index + k];
                }
            }

            for (int j_block = 0; j_block < size; j_block += block_size) {

                double* b_pack = new double[block_size * block_size];
                for (int k = 0; k < block_size; k++) {
                    const int b_index = (k_block + k) * size + j_block;
                    for (int j = 0; j < block_size; j++) {
                        b_pack[k * block_size + j] = matrix_b[b_index + j];
                    }
                }

                const int full_block_size = block_size*block_size;    
                for (int index = 0; index < full_block_size; index++) {
                    const int i = (i_block + index / block_size) * size + j_block;
                    const int k = (index % block_size) * block_size;
                    const double a_value = a_pack[index];
                    for (int j = 0; j < block_size; j++) {
                        result_matrix[i + j] += a_value * b_pack[k + j];
                    }
                }

                delete[] b_pack;
            }
            
            delete[] a_pack;
        } 
    }  
}

void multiply_ikj_arr_packing2(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    for (int i_block = 0; i_block < size; i_block += block_size) {
        for (int k_block = 0; k_block < size; k_block += block_size) {

            double* a_pack = new double[block_size * block_size];
            for (int i = 0; i < block_size; i++) {
                const int a_index = (i_block + i) * size + k_block;
                for (int k = 0; k < block_size; k++) {
                    a_pack[i * block_size + k] = matrix_a[a_index + k];
                }
            }

            for (int j_block = 0; j_block < size; j_block += block_size) {

                double* b_pack = new double[block_size * block_size];
                for (int k = 0; k < block_size; k++) {
                    const int b_index = (k_block + k) * size + j_block;
                    for (int j = 0; j < block_size; j++) {
                        b_pack[k * block_size + j] = matrix_b[b_index + j];
                    }
                }

                double* c_pack = new double[block_size * block_size];
                for (int i = 0; i < block_size; i++) {
                    const int c_index = (i_block + i) * size + j_block;
                    for (int j = 0; j < block_size; j++) {
                        c_pack[i * block_size + j] = result_matrix[c_index + j];
                    }
                }

                const int full_block_size = block_size*block_size;    
                for (int index = 0; index < full_block_size; index++) {
                    const int i = (index / block_size) * block_size;
                    const int k = (index % block_size) * block_size;
                    const double a_value = a_pack[index];
                    for (int j = 0; j < block_size; j++) {
                        c_pack[i + j] += a_value * b_pack[k + j];
                    }
                }

                delete[] b_pack;

                for (int i = 0; i < block_size; i++) {
                    const int c_index = (i_block + i) * size + j_block;
                    for (int j = 0; j < block_size; j++) {
                        result_matrix[c_index + j] = c_pack[i * block_size + j];
                    }
                }

                delete[] c_pack;
            }
            
            delete[] a_pack;
        } 
    }  
}

void multiply_ikj_arr_blocking_non_square(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                          const int size, const int i_block_size, const int j_block_size, const int k_block_size) {
    for (int i_block = 0; i_block < size; i_block += i_block_size) {
        for (int k_block = 0; k_block < size; k_block += k_block_size) {
            for (int j_block = 0; j_block < size; j_block += j_block_size) {
                const int full_block_size = i_block_size * k_block_size;    
                for (int index = 0; index < full_block_size; index++) {
                    const int i = (i_block + index / k_block_size) * size;
                    const int k = k_block + index % k_block_size;
                    const double a_value = matrix_a[i + k];
                    const int result_index = i + j_block;
                    const int b_index = k * size + j_block;
                    for (int j = 0; j < j_block_size; j++) {
                        result_matrix[result_index + j] += a_value * matrix_b[b_index + j];
                    }
                }
            } 
        } 
    }  
}

void multiply_ikj_arr_packing_non_square(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                         const int size, const int i_block_size, const int j_block_size, const int k_block_size) {
    for (int i_block = 0; i_block < size; i_block += i_block_size) {
        for (int k_block = 0; k_block < size; k_block += k_block_size) {

            double* a_pack = new double[i_block_size * k_block_size];
            for (int i = 0; i < i_block_size; i++) {
                const int a_index = (i_block + i) * size + k_block;
                for (int k = 0; k < k_block_size; k++) {
                    a_pack[i * k_block_size + k] = matrix_a[a_index + k];
                }
            }

            for (int j_block = 0; j_block < size; j_block += j_block_size) {

                double* b_pack = new double[k_block_size * j_block_size];
                for (int k = 0; k < k_block_size; k++) {
                    const int b_index = (k_block + k) * size + j_block;
                    for (int j = 0; j < j_block_size; j++) {
                        b_pack[k * j_block_size + j] = matrix_b[b_index + j];
                    }
                }

                const int full_block_size = i_block_size * k_block_size;    
                for (int index = 0; index < full_block_size; index++) {
                    const int i = (i_block + index / k_block_size) * size + j_block;
                    const int k = (index % k_block_size) * j_block_size;
                    const double a_value = a_pack[index];
                    for (int j = 0; j < j_block_size; j++) {
                        result_matrix[i + j] += a_value * b_pack[k + j];
                    }
                }

                delete[] b_pack;
            }
            
            delete[] a_pack;
        } 
    }  
}





// O2

void multiply_ikj_arr_recursive_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                const int i_offset, const int j_offset, const int k_offset, 
                                const int size, const int full_size, const int block_size) {
    if (size <= block_size) {  
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                const double a_value = matrix_a[(i_offset + i) * full_size + (k_offset + k)];
                for (int j = 0; j < block_size; j++) {
                    result_matrix[(i_offset + i) * full_size + (j_offset + j)] += a_value * matrix_b[(k_offset + k) * full_size + (j_offset + j)];
                }
            }
        }
    } else {
        const int new_size = size / 2;
        const int new_i_offset = i_offset + new_size;
        const int new_j_offset = j_offset + new_size;
        const int new_k_offset = k_offset + new_size;

        // ikj
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, i_offset, j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, i_offset, new_j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, i_offset, j_offset, new_k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, i_offset, new_j_offset, new_k_offset, new_size, full_size, block_size);

        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, new_i_offset, j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, new_i_offset, new_j_offset, k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, new_i_offset, j_offset, new_k_offset, new_size, full_size, block_size);
        multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, new_i_offset, new_j_offset, new_k_offset, new_size, full_size, block_size);
    }
}

void multiply_ikj_arr_recursive_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
    multiply_ikj_arr_recursive_simple(matrix_a, matrix_b, result_matrix, 0, 0, 0, size, size, block_size);
}


void multiply_ikj_arr_packing_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
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

void multiply_ikj_arr_packing2_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, const int size, const int block_size) {
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

                double* c_pack = new double[block_size * block_size];
                for (int i = 0; i < block_size; i++) {
                    for (int j = 0; j < block_size; j++) {
                        c_pack[i * block_size + j] = result_matrix[(i_block + i) * size + j_block + j];
                    }
                }

                for (int i = 0; i < block_size; i++) {
                    for (int k = 0; k < block_size; k++) {
                        const double a_value = a_pack[i * block_size + k];
                        for (int j = 0; j < block_size; j++) {
                            c_pack[i * block_size + j] += a_value * b_pack[k * block_size + j];
                        }
                    }
                }

                delete[] b_pack;

                for (int i = 0; i < block_size; i++) {
                    for (int j = 0; j < block_size; j++) {
                        result_matrix[(i_block + i) * size + j_block + j] = c_pack[i * block_size + j];
                    }
                }

                delete[] c_pack;
            }
            
            delete[] a_pack;
        } 
    }  
}


void multiply_ikj_arr_blocking_non_square_simple(const double* matrix_a, const double* matrix_b, double* result_matrix, 
                                          const int size, const int i_block_size, const int j_block_size, const int k_block_size) {
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
    const int BLOCK_SIZE = 512;

    const int I_BLOCK_SIZE = 64;
    const int J_BLOCK_SIZE = 2048;
    const int K_BLOCK_SIZE = 8;
    
    
    double* matrix_a = generate_matrix_arr(N);
    double* matrix_b = generate_matrix_arr(N);

    double* result_matrix = new double[N * N]();

    auto start_time = std::chrono::steady_clock::now();
    multiply_ikj_arr_packing_non_square_simple(matrix_a, matrix_b, result_matrix, N, I_BLOCK_SIZE, J_BLOCK_SIZE, K_BLOCK_SIZE);
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

    for (int i = 2; i <= 2048; i *= 2) {
        for (int j = 2048; j <= 2048; j *= 2) {
            for (int k = 2; k <= 2048; k *= 2) {
                double* matrix_a = generate_matrix_arr(2048);
                double* matrix_b = generate_matrix_arr(2048);

                double* result_matrix = new double[2048 * 2048]();

                auto start_time = std::chrono::steady_clock::now();
                multiply_ikj_arr_blocking_non_square_simple(matrix_a, matrix_b, result_matrix, 2048, i, j, k);
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