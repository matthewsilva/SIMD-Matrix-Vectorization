#!/bin/bash
g++ matvec_bash.cc -o matvec_bash -std=gnu++11 -mavx -I /usr/include/eigen3
g++ matvec_bash.cc -o matvec_bash_O2 -std=gnu++11 -mavx -I /usr/include/eigen3 -O2
g++ matvec_bash.cc -o matvec_bash_O3 -std=gnu++11 -mavx -I /usr/include/eigen3 -O3
g++ simd_matvec_bash.cc -o simd_matvec_bash -std=gnu++11 -mavx -I /usr/include/eigen3
