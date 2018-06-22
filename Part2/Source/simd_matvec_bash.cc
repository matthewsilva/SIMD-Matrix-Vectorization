// Author :	Matthew Silva
// 			CSC 415
//			Professor Alvarez
//			URI
//			12 March 2018


#include <iostream>
#include <unistd.h>
#include <immintrin.h>
#include <chrono>
#include <Eigen/Dense>


// Sequential implementation of a matrix vectorization
void matvec(const float *A, const float *x, unsigned int m, unsigned int n, float *b) {
	float vecSum;
	unsigned int in;
	for (unsigned int i = 0; i < m; i++) {
		vecSum = 0; // Reset sum to 0 after every row
		in = i*n;
		for (unsigned int j = 0; j < n; j++) {
			// Sum each product of corresponding matrix and vector values
			vecSum += (A[in + j] * x[j]);
		}
		// Record the sum in the resulting vector
		b[i] = vecSum;
	}
}

// SIMD implementation of a matrix vectorization
void matvec_simd(const float *A, const float *x, unsigned int m, unsigned int n, float *b) {
		
	float vecSum;
	// Find how many floats can fit into a 256 bit register	
	unsigned int regLen = (256/8) / sizeof(float);
	
	// Allocate an memory-aligned (aligned by 32 bytes) float array the size of a register
	float *sumRegArr = (float *) _mm_malloc(regLen * sizeof(float), 32);
	
	// Three 256 bit registers
	__m256 reg0, reg1, reg2;
	// Intialize reg2 to all 0's
	reg2 = _mm256_setzero_ps();

	unsigned int in;

	// How many full iterations can we go through 
	// collecting one register-full of values each iteration?
	int n_iter = n / regLen;
	// What is the starting column index of the last full iteration?
	int nMaxVecIndex = n - (n % regLen);

	// If there are only full iterations (n is divisible by register length),
	// we can handle the whole process using SIMD operations on registers
	if (n % regLen == 0) {
		for (unsigned int i = 0; i < m; i++) {

			vecSum = 0; // Reset sum to 0 after every row
			in = i*n; // Code motion
			reg2 = _mm256_setzero_ps(); // Reset reg2 to all 0's after every row

			for (unsigned int j = 0; j < n; j += regLen) {

				// Load reg0 and reg1 with the next 8 float values of the matrix and vector 
				reg0 = _mm256_load_ps(A + in + j);
				reg1 = _mm256_load_ps(x + j);

				// Multiply reg0 and reg1 and accumulate the result in reg2
				reg0 = _mm256_mul_ps(reg0, reg1);			
				reg2 = _mm256_add_ps(reg0, reg2);
			
			}
			// After the row is done, store reg2 into an array
			_mm256_store_ps(sumRegArr, reg2);
			for (unsigned int k = 0; k < regLen; k++) {
				vecSum += sumRegArr[k]; // Sum the register values of reg2 using the array
			}

			b[i] = vecSum; // Store the row's sum into a result vector
		
		}
		_mm_free(sumRegArr);
		}
	// Else, there are extra values that must be handled sequentially
	else {
		for (unsigned int i = 0; i < m; i++) {
			vecSum = 0; // Reset sum to 0 after every row
			in = i*n; // Code motion
			reg2 = _mm256_setzero_ps(); // Reset reg2 to all 0's after every row

			// Only go up to the last full iteration
			for (unsigned int j = 0; j < nMaxVecIndex; j += regLen) {

				// Load reg0 and reg1 with the next 8 float values of the matrix and vector 
				reg0 = _mm256_loadu_ps(A + in + j);
				reg1 = _mm256_loadu_ps(x + j);

				// Multiply reg0 and reg1 and accumulate the result in reg2
				reg0 = _mm256_mul_ps(reg0, reg1);			
				reg2 = _mm256_add_ps(reg0, reg2);
			
			
			}
			// Process the data outside of a full iteration sequentially
			for (unsigned int k = nMaxVecIndex; k < n; k++) {
				
				// Sum each product of corresponding matrix and vector values
				vecSum += A[in + k] * x[k]; 
			}

			// Now unload the sum of the full iterations...
			_mm256_store_ps(sumRegArr, reg2);
			for (unsigned int k = 0; k < regLen; k++) {
				vecSum += sumRegArr[k]; // Sum the register values of reg2 using the array
			}

			b[i] = vecSum; // Store the row's sum into a result vector
		
		}
		_mm_free(sumRegArr);
	}
}

// SIMD implementation of a matrix vectorization with loop unrolling
void matvec_simd_unrolled(const float *A, const float *x, unsigned int m, unsigned int n, float *b) {
		
	float totalSum;	
	
	// Four separate accumulators
	float vecSum_0;
	float vecSum_1;
	float vecSum_2;
	float vecSum_3;
	// One accumulator for sequential operations
	float seqVecSum;

	// Find how many floats can fit into a 256 bit register	
	unsigned int regLen = (256/8) / sizeof(float);

	// Allocate four memory-aligned (aligned by 32 bytes) float arrays the size of a register
	float *sumRegArr_0 = (float *) _mm_malloc(regLen * sizeof(float), 32);
	float *sumRegArr_1 = (float *) _mm_malloc(regLen * sizeof(float), 32);
	float *sumRegArr_2 = (float *) _mm_malloc(regLen * sizeof(float), 32);
	float *sumRegArr_3 = (float *) _mm_malloc(regLen * sizeof(float), 32);

	// Four separate sets of registers
	__m256 reg0_0, reg1_0, reg2_0;
	__m256 reg0_1, reg1_1, reg2_1;
	__m256 reg0_2, reg1_2, reg2_2;
	__m256 reg0_3, reg1_3, reg2_3;

	// Intialize reg2s to all 0's
	reg2_0 = _mm256_setzero_ps();
	reg2_1 = _mm256_setzero_ps();
	reg2_2 = _mm256_setzero_ps();
	reg2_3 = _mm256_setzero_ps();
	
	// Common subexpressions / code motion
	unsigned int in;
	int inj_0;
	int inj_1;
	int inj_2;
	int inj_3;
	
	// Unrolling variables
	unsigned const int rolls = 4;
	unsigned int regLenRolls = regLen*rolls;
	int nMaxVecIndex = n - (n % (4*regLen));

	// If there are only full iterations (n is divisible by FOUR times the register length),
	// we can handle the whole process using SIMD operations on registers
	if ((n % (regLen*4)) == 0) {
		for (unsigned int i = 0; i < m; i++) {
			
			// Reset all accumulators to zero after a row			
			vecSum_0 = 0;
			vecSum_1 = 0;
			vecSum_2 = 0;
			vecSum_3 = 0;
			
			in = i*n; // code motion
	
			// Reset all reg2s to all 0's after a row
			reg2_0 = _mm256_setzero_ps();
			reg2_1 = _mm256_setzero_ps();
			reg2_2 = _mm256_setzero_ps();
			reg2_3 = _mm256_setzero_ps();

			// Iterate by four times the register length
			for (unsigned int j = 0; j < nMaxVecIndex; j += regLenRolls) {
				
				inj_0 = in + j;
				
				// Load reg0 and reg1 with the next 8 float values of the matrix and vector
				reg0_0 = _mm256_load_ps(A + inj_0);
				reg1_0 = _mm256_load_ps(x + j);
				
				// Multiply reg0 and reg1 and accumulate the result in reg2
				reg0_0 = _mm256_mul_ps(reg0_0, reg1_0);			
				reg2_0 = _mm256_add_ps(reg0_0, reg2_0);

				// Do the above three more times, moving forward by 1 register length each time
				inj_1 = in + j + regLen;
				reg0_1 = _mm256_load_ps(A + inj_1);
				reg1_1 = _mm256_load_ps(x + j + regLen);
				reg0_1 = _mm256_mul_ps(reg0_1, reg1_1);			
				reg2_1 = _mm256_add_ps(reg0_1, reg2_1);
				inj_2 = in + j + 2*regLen;
				reg0_2 = _mm256_load_ps(A + inj_2);
				reg1_2 = _mm256_load_ps(x + j + 2*regLen);
				reg0_2 = _mm256_mul_ps(reg0_2, reg1_2);			
				reg2_2 = _mm256_add_ps(reg0_2, reg2_2);
				inj_3 = in + j + 3*regLen;
				reg0_3 = _mm256_load_ps(A + inj_3);
				reg1_3 = _mm256_load_ps(x + j + 3*regLen);
				reg0_3 = _mm256_mul_ps(reg0_3, reg1_3);			
				reg2_3 = _mm256_add_ps(reg0_3, reg2_3);
			
			
			}
			// Now unload the four registers into arrays...
			_mm256_store_ps(sumRegArr_0, reg2_0);
			_mm256_store_ps(sumRegArr_1, reg2_1);
			_mm256_store_ps(sumRegArr_2, reg2_2);
			_mm256_store_ps(sumRegArr_3, reg2_3);
		
			for (unsigned int k = 0; k < regLen; k++) {
				// Sum each register's values using the array
				vecSum_0 += sumRegArr_0[k]; 
				vecSum_1 += sumRegArr_1[k]; 
				vecSum_2 += sumRegArr_2[k]; 
				vecSum_3 += sumRegArr_3[k]; 
			}
			// Accumulate each register sum into a total
			totalSum = vecSum_0 + vecSum_1 + vecSum_2 + vecSum_3;

			// Store the total in the result vector
			b[i] = totalSum;
		
		}
	}
	else {

		for (unsigned int i = 0; i < m; i++) {

			// Reset all accumulators to zero after a row			
			vecSum_0 = 0;
			vecSum_1 = 0;
			vecSum_2 = 0;
			vecSum_3 = 0;
			seqVecSum = 0;

			in = i*n; // code motion

			// Reset all reg2s to all 0's after a row
			reg2_0 = _mm256_setzero_ps();
			reg2_1 = _mm256_setzero_ps();
			reg2_2 = _mm256_setzero_ps();
			reg2_3 = _mm256_setzero_ps();

			// Iterate by four times the register length up to the last full iteration
			for (unsigned int j = 0; j < nMaxVecIndex; j += regLenRolls) {
				inj_0 = in + j;

				// Load reg0 and reg1 with the next 8 float values of the matrix and vector 
				reg0_0 = _mm256_loadu_ps(A + inj_0);
				reg1_0 = _mm256_loadu_ps(x + j);

				// Multiply reg0 and reg1 and accumulate the result in reg2
				reg0_0 = _mm256_mul_ps(reg0_0, reg1_0);			
				reg2_0 = _mm256_add_ps(reg0_0, reg2_0);
				
				// Do the above three more times, moving forward by 1 register length each time
				inj_1 = in + j + regLen;
				reg0_1 = _mm256_loadu_ps(A + inj_1);
				reg1_1 = _mm256_loadu_ps(x + j + regLen);
				reg0_1 = _mm256_mul_ps(reg0_1, reg1_1);			
				reg2_1 = _mm256_add_ps(reg0_1, reg2_1);
				inj_2 = in + j + 2*regLen;
				reg0_2 = _mm256_loadu_ps(A + inj_2);
				reg1_2 = _mm256_loadu_ps(x + j + 2*regLen);
				reg0_2 = _mm256_mul_ps(reg0_2, reg1_2);			
				reg2_2 = _mm256_add_ps(reg0_2, reg2_2);
				inj_3 = in + j + 3*regLen;
				reg0_3 = _mm256_loadu_ps(A + inj_3);
				reg1_3 = _mm256_loadu_ps(x + j + 3*regLen);
				reg0_3 = _mm256_mul_ps(reg0_3, reg1_3);			
				reg2_3 = _mm256_add_ps(reg0_3, reg2_3);
			
			
			}
			// Handle the rest of the data sequentially
			for (unsigned int k = nMaxVecIndex; k < n; k++) {
				seqVecSum += A[in + k] * x[k]; // Sum the rest of the products sequentially
			}

			// Now unload the four registers into arrays...
			_mm256_store_ps(sumRegArr_0, reg2_0);
			_mm256_store_ps(sumRegArr_1, reg2_1);
			_mm256_store_ps(sumRegArr_2, reg2_2);
			_mm256_store_ps(sumRegArr_3, reg2_3);
			for (unsigned int k = 0; k < regLen; k++) {

				// Sum each register's values using the array
				vecSum_0 += sumRegArr_0[k]; 
				vecSum_1 += sumRegArr_1[k]; 
				vecSum_2 += sumRegArr_2[k]; 
				vecSum_3 += sumRegArr_3[k]; 
			}
			// Accumulate each register sum and the sequential sum into a total
			totalSum = vecSum_0 + vecSum_1 + vecSum_2 + vecSum_3 + seqVecSum;

			// Store the total in the result vector
			b[i] = totalSum;
		
		}
	}
	// Free all the memory aligned arrays
	_mm_free(sumRegArr_0);
	_mm_free(sumRegArr_1);
	_mm_free(sumRegArr_2);
	_mm_free(sumRegArr_3);

}



int main(int argc, char *argv[]) {

	// m (# of rows) is first argument
	unsigned int m;
	sscanf (argv[1],"%d",&m);
	// n (# of columns) is second argument
	unsigned int n;
	sscanf (argv[2],"%d",&n);

	// Allocate memory-aligned (aligned along 32 byte bounds) float arrays for use in vectorization 
    float *A = (float *) _mm_malloc(m * n * sizeof(float), 32);
    float *x = (float *) _mm_malloc(n * 1 * sizeof(float), 32);
	float *b0 = (float *) _mm_malloc(m * sizeof(float), 32);
	float *b1 = (float *) _mm_malloc(m * sizeof(float), 32);
    float *b2 = (float *) _mm_malloc(m * sizeof(float), 32);

	// Set A to random values between -500 and 500
	srand (static_cast <unsigned> (time(0)));
	for (unsigned int i = 0 ; i < m*n ; i++) {
        A[i] = -500 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1000)));
    }
	// Set x to random values between -500 and 500
	for (unsigned int i = 0 ; i < n*1 ; i++) {
        x[i] = -500 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1000)));
    }
	
	// First column of CSV shows matrix dimensions
	std::cout << m << "*" << n << " matrix,";

	// 2nd Column of CSV: Time of execution of matvec() on A and x to b0
	auto start0 = std::chrono::high_resolution_clock::now();
	matvec(A, x, m, n, b0);
	auto end0 = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> diff0 = end0 - start0;
    std::cout << diff0.count() << " sec,";

	// 3rd Column of CSV: Time of execution of matvec_simd() on A and x to b1
	auto start1 = std::chrono::high_resolution_clock::now();
	matvec_simd(A, x, m, n, b1);
	auto end1 = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> diff1 = end1 - start1;
    std::cout<< diff1.count() << " sec,";
	
	// 4th Column of CSV: Time of execution of matvec_simd_unrolled() on A and x to b1
	auto start2 = std::chrono::high_resolution_clock::now();
	matvec_simd_unrolled(A, x, m, n, b2);
	auto end2 = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> diff2 = end2 - start2;
    std::cout << diff2.count() << " sec,";

	// 5th Column of CSV: Speedup from matvec() to matvec_simd()
	float seqToVecspeedUp = diff0.count() / diff1.count();
	std::cout << seqToVecspeedUp << ",";

	// 6th Column of CSV: Speedup from matvec() to matvec_simd_unrolled()
	float seqToUnrollspeedUp = diff0.count() / diff2.count();
	std::cout << seqToUnrollspeedUp << ",";

	// 7th Column of CSV: Speedup from matvec_simd() to matvec_simd_unrolled()
	float vecToUnrollspeedUp = diff1.count() / diff2.count();
	std::cout << vecToUnrollspeedUp << "\n";
	
	
	/* 
		Commented out for CSV use
		
	// Using Eigen to verify matrix vectorization is correct 
	// (Code on Piazza, thank you Doug!)

	Eigen::MatrixXf eigenMat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Aligned>(A, m, n);
	Eigen::MatrixXf eigenVec = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Aligned>(x, n, 1);
	
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenAns = eigenMat * eigenVec;
	// Correct answer for vectorization from Eigen
	float* answer = eigenAns.data();
	
	// Verify sequential vectorization against Eigen's answer
	bool matVecWrong = false;
	for (unsigned int i = 0 ; i < m*1 && !matVecWrong ; i++) {
        if (abs(b0[i] - answer[i]) > 2.5) {
			std::cout << "Sequential vectorization Incorrect for " << m << " by " << n << " matrix: " << b0[i] - answer[i] << " off \n";
			matVecWrong = true;
		}
    }

	// Verify SIMD vectorization against Eigen's answer
	bool simdVecWrong = false;
	for (unsigned int i = 0 ; i < m*1 && !simdVecWrong ; i++) {
        if (abs(b1[i] - answer[i]) > 2.5) {
			std::cout << "SIMD vectorization Incorrect for " << m << " by " <<  n << " matrix: " << b1[i] - answer[i] << " off \n";
			simdVecWrong = true;
		}
    }
	
	// Verify unrolled SIMD vectorization against Eigen's answer
	bool simdUnrollVecWrong = false;
	for (unsigned int i = 0 ; i < m*1 && !simdUnrollVecWrong ; i++) {
        if (abs(b2[i] - answer[i]) > 2.5) {
			std::cout << "SIMD unrolled vectorization Incorrect for " << m << " by " << n << " matrix: " << b2[i] - answer[i] << " off \n";
			simdUnrollVecWrong = true;
		}
    }
	*/
	return 0;
}







