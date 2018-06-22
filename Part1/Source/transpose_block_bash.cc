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

void transpose_block(const int *A, unsigned int m, unsigned int n, unsigned int block_size, int *AT) {
    
	unsigned int i, j, k, l, kn;
	
	// Start at 0 and iterate by block size in outer loops to handle each block once
    for (i = 0 ; i < m ; i += block_size ) {
        for (j = 0 ; j < n ; j += block_size ) {
			// Abstractly, these inner loops are a procedure to handle one block
            for (k = i ; k < i + block_size && k < m ; k++) { 
				kn = k*n; //  Code motion
        		for (l = j ; l < j + block_size && l < n ; l++) {
            		// Transpose one value from A to AT
					AT[k+l*m] = A[l+kn];
        		}
    		}
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
	// Block size of block caching transposition is third argument
	unsigned int blockSize;
	sscanf (argv[3],"%d",&blockSize);
	
	// A is an m*n matrix, and it's transpose AT is n*m
	int *A = new int [m * n];
    int *AT = new int [n * m];
	
	// Intialize A with random values between -500 and 500
    std::srand(std::time(nullptr));
    for (unsigned int i = 0 ; i < m*n ; i++) {
        A[i] = -500 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1000)));
    }
	// First column of CSV shows matrix dimensions
	std::cout << m << " * " << n << " (" << blockSize << " blockSize),";
	
	// 2nd Column of CSV: Time of execution of transpose() on A to AT
	auto start0 = std::chrono::high_resolution_clock::now();
	transpose(A, m, n, AT);
    auto end0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff0 = end0 - start0;
    std::cout << diff0.count() << ",";
	
	// 3rd Column of CSV: Time of execution of transpose_block() on A to AT
    auto start1 = std::chrono::high_resolution_clock::now();
	transpose_block(A, m, n, blockSize, AT);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = end1 - start1;
    std::cout << diff1.count() << ",";
	
	// 4th Column of CSV: Speedup from transpose() to transpose_block()
	float seqToBlockSpeedUp = diff0.count() / diff1.count();
	std::cout << seqToBlockSpeedUp << "\n";
	
	// Check whether transpose_block correctly transposed A to AT
    if (! check_transpose(A, AT, n, m)) {
        std::cout << "Incorrect !";
    }
	
    delete [] A;
    delete [] AT;
}
