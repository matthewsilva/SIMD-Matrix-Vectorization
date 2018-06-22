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

int main(int argc, char *argv[]) {

	unsigned int m;
	sscanf (argv[1],"%d",&m);
	unsigned int n;
	sscanf (argv[2],"%d",&n);

    float *A = new float [m * n];
    float *x = new float [n * 1];
	float *b1 = new float [m * 1];
	
	srand (static_cast <unsigned> (time(0)));
	for (unsigned int i = 0 ; i < m*n ; i++) {
        A[i] = -500 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1000)));
    }
	for (unsigned int i = 0 ; i < n*1 ; i++) {
        x[i] = -500 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1000)));
    }

	char hostname[512];
	gethostname(hostname, 512);
	
	auto start = std::chrono::high_resolution_clock::now();
	matvec(A, x, m, n, b1);
	auto end = std::chrono::high_resolution_clock::now();	
	std::chrono::duration<float> diff = end - start;
    std::cout << diff.count() << ",";

	/*
			Commented out for CSV use
		

	// Using Eigen to verify matrix vectorization is correct 
	// (Code on Piazza, thank you Doug!)
	
	Eigen::MatrixXf eigenMat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Aligned>(A, m, n);
	Eigen::MatrixXf eigenVec = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Aligned>(x, n, 1);
	
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenAns = eigenMat * eigenVec;
	// Correct answer for vectorization from Eigen
	float* answer = eigenAns.data();
	
	// Verify vectorization against Eigen's answer
	bool stdVecWrong = false;	
	for (unsigned int i = 0 ; i < m*1 && !stdVecWrong ; i++) {
        if (abs(b1[i] - answer[i]) > 0.5) {
			std::cout << "Standard vectorization Incorrect: " << b1[i] - answer[i] << " off \n";
			stdVecWrong = true;			
		}
    }
	*/
	return 0;
}







