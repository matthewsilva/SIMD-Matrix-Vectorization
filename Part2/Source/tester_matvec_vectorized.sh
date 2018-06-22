#!/bin/bash
#
#SBATCH --job-name=matvec_vectorized
#SBATCH --error=errorVectorized.txt
#SBATCH --output=outputVectorized.txt
#
#SBATCH --ntasks=1
#SBATCH --time=4:00
#SBATCH --mem-per-cpu=20gb
#
#SBATCH --constraint=gpu

echo "Dimensions,SEQtime,VECtime,UNROLLtime,SEQtoVECSpeedup,SEQtoUNROLL,VECtoUNROLL"
for ((i=1; i< 32768; i = i * 8))
	do
		for ((j=1; j< 32768; j = j * 8))
			do
				./simd_matvec_bash $i $j
			done
	done
for ((i=1; i< 16807; i = i * 7))
	do
		for ((j=1; j< 16807; j = j * 7))
			do
				./simd_matvec_bash $i $j
			done
	done

