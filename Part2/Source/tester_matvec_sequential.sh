#!/bin/bash
#
#SBATCH --job-name=matvec_sequential
#SBATCH --error=errorSequential.txt
#SBATCH --output=outputSequential.txt
#
#SBATCH --ntasks=1
#SBATCH --time=4:00
#SBATCH --mem-per-cpu=20gb
#
#SBATCH --constraint=gpu

echo "Dimensions,Baseline(sec),-O2(sec),-O3(sec),-O2Speedup,-O3Speedup,-O2to-O3Speedup"
for ((i=1; i< 32768; i = i * 8))
	do
		for ((j=1; j< 32768; j = j * 8))
			do
				printf "$i * $j matrix,"
				./matvec_bash $i $j
				./matvec_bash_O2 $i $j
				./matvec_bash_O3 $i $j
				printf "\n"
			done
	done
for ((i=1; i< 16807; i = i * 7))
	do
		for ((j=1; j< 16807; j = j * 7))
			do
				printf "$i * $j matrix,"
				./matvec_bash $i $j
				./matvec_bash_O2 $i $j
				./matvec_bash_O3 $i $j
				printf "\n"
			done
	done

