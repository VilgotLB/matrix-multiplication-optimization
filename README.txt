PROBLEM 1:

Compile:
g++ p1/problem1.cpp -o p1/problem1

Run:
p1/problem1

Change matrix size by adjusting N in main

-----------------------------------------------------

PROBLEM 2

Instructions vary depending on which implementation is used
* Matmul function selected on line 263
* Matrix size changed by adjusting N in main

For the first six (matmul_ijk, matmul_jik, matmul_ikj, matmul_kij, matmul_jki, matmul_kji):

Compile: g++ p2/problem2.cpp -march=native -o p2/problem2
Run: p2/problem2

For matmul_ikj_arr and below:

* Change generate_matrix to generate_matrix_arr for both A and B in main
* Change generate_empty to generate_empty_arr in main
* Uncomment lines 271-273 to deallocate the arrays after use

For matmul_ikj_arr_blocking and below:

* Enable -O2 optimization
    * Compile: g++ p2/problem2.cpp -O2 -march=native -o p2/problem2
    * Run: p2/problem2
* Block size adjusted with BLOCK_SIZE in main
* Include BLOCK_SIZE as an argument to the matmul function

For matmul_ikj_arr_blocking_vector_aligned and below:

* Set align to true in generate_matrix_arr and generate_empty_arr

Verify correctness:
* Uncomment the test line in main (line 268)
* Outputs the L2 norm of the difference matrix between my implementation and the Eigen library implementation. Close to 0 means correct.
* Only works for matmul_ikj_arr and below

Compare with numpy:
Run test.py. Keep line 2 uncommented.

----------------------------------------------

PROBLEM 3:

Adjust number of threads with omp_set_num_threads() in main.
Adjust matrix size with N in main.
Adjust block size with BLOCK_SIZE in main.
Change function on line 162.

Compile: g++ p3/problem3.cpp -fopenmp -O2 -march=native -o p3/problem3
Run: p3/problem3

Verify correctness:
Uncomment the test line on line 168

Compare with numpy:
Comment out line 2 in test.py and run
