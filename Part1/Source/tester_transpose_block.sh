#!/bin/bash
#
#SBATCH --job-name=transpose_block
#SBATCH --error=errorBlock.txt
#SBATCH --output=outputBlock.txt
#
#SBATCH --ntasks=1
#SBATCH --time=4:00
#SBATCH --mem-per-cpu=20gb
#
#SBATCH --constraint=gpu

echo "Dimensions,SEQtime,BLOCKtime,SEQtoBLOCKSpeedup"
for i in 2 50 100 250 500 1000 2500 13
	do
	   ./transpose_block_bash 5000 5000 $i
	done
for i in 3 51 101 251 501 1001 2501
	do
	   ./transpose_block_bash 5000 5000 $i
	done
