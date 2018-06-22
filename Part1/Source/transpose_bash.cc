// Author :	Matthew Silva
// 			CSC 415
//			Professor Alvarez
//			URI
//			12 March 2018


#include <iostream>
#include <chrono>

// input matrix A (row-major order) of dimensions m x n
// output matrix AT (row-major order) of dimensions n x m
void transpose(const int *A, unsigned int m, unsigned int n, int *AT) {
    unsigned int i, j;

    for (i = 0 ; i < m ; i++ ) {
        for (j = 0 ; j < n ; j++ ) {
            AT[i+j*m] = A[j+i*n];
        }
    }
}


bool check_transpose(const int *A, const int *AT, unsigned int m, unsigned int n) {
    unsigned int i, j;

    for ( i = 0 ; i < m ; i++ ) {
        for ( j = 0 ; j < n ; j++ ) {
            if ( AT[j+i*n] != A[i+j*m] ) {
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char *argv[]) {
    
	// m (# of rows) is first argument
	unsigned int m;
	sscanf (argv[1],"%d",&m);
	// n (# of columns) is second argument
	unsigned int n;
	sscanf (argv[2],"%d",&n);
	
	// A is an m*n matrix, and it's transpose AT is n*m
	int *A = new int [m * n];
    int *AT = new int [n * m];
	
	// Intialize A with random values between -500 and 500
    std::srand(std::time(nullptr));
    for (unsigned int i = 0 ; i < m*n ; i++) {
        A[i] = -500 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1000)));
    }

	// 2nd Column of CSV: Time of execution of transpose() on A to AT
    auto start = std::chrono::high_resolution_clock::now();    
	transpose(A, m, n, AT);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count() << ",";

	// Check whether transpose() correctly transposed A to AT
    if (! check_transpose(A, AT, n, m)) {
        std::cout << "Incorrect !";
    }

    delete [] A;
    delete [] AT;
}
