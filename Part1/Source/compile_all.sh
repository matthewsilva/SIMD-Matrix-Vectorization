#!/bin/bash
g++ transpose_bash.cc -o transpose_bash -std=gnu++11
g++ transpose_bash.cc -o transpose_bash_O2 -std=gnu++11 -O2
g++ transpose_bash.cc -o transpose_bash_O3 -std=gnu++11 -O3
g++ transpose_block_bash.cc -o transpose_block_bash -std=gnu++11
