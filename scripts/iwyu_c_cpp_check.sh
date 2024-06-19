#! /usr/bin/env bash

# This script wraps the make command for running the IWYU tests native to cmake.
# It takes one positional argument. The top level directory of NVSHMEM.
# Internally, it calls a python script in this same directory to parse and output
# failure information.
# Exits 0 on pass, 1 on fail.

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_nvshmem>"
    exit 1
fi

cmake -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE="include-what-you-use" -DCMAKE_C_INCLUDE_WHAT_YOU_USE="include-what-you-use" -DNVSHMEM_BUILD_TESTS=OFF -DNVSHMEM_BUILD_EXAMPLES=OFF -DNVSHMEM_BUILD_PACKAGES=OFF -S $1 -B $1/build
make -C $1/build -j8 > $1/logs.txt 2>&1 || true
if cat $1/logs.txt | grep -q "include-what-you-use reported diagnostics"; then
    echo "include-what-you-use reported errors. Build failing."
    ${CI_PROJECT_DIR}/scripts/iwyu_output_parser.py $1/logs.txt
    exit 1
fi

exit 0