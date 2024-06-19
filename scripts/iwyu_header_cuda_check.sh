#! /usr/bin/env bash

# This script forces IWYU to check both .h and .cuh headers as well as .cu files
# This is not supported natively in cmake so we construct the command locally and
# run IWYU directly.
# It takes two positional arguments NVSHMEM source top level directory and CUDA install path.
# Exits 0 when no include errors are found, 1 otherwise.

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_nvshmem> <path_to_cuda>"
    exit 1
fi

nvshmem_home=$1
cuda_home=$2

exit_code=0
formatting_string="\n______________________________________________________________________________________\n"

#filenames used by this script
script_tld=/tmp/`uuidgen`
nvshmem_header_directories_file="$script_tld/nvshmem_include_dirnames.txt"
nvshmem_header_file="$script_tld/nvshmem_include_files.txt"
nvshmem_cuda_file="$script_tld/nvshmem_cuda_files.txt"
nvshmem_c_cpp_file="$script_tld/nvshmem_c_cpp_files.txt"

# these arguments are needed to properly run headers files through IWYU
# You can check man clang for the meaning of these flags
iwyu_command_prefix="iwyu "
iwyu_command_suffix=" -isystem $2/include --std=c++11 -xc++ -D__CUDACC__ -D__CUDA_ARCH__"
iwyu_command="$iwyu_command_prefix"

detect_failures() {
    error_count=`wc -l < "$1"`
    if [ "$error_count" -ne "0" ]; then
        echo -e $formatting_string
        echo "error: $2"
        cat $1
        echo -e $formatting_string
        return 1
    fi
    return 0
}

check_iwyu_quirks() {
    nvshmem_build_options_file=$nvshmem_home/src/include/non_abi/nvshmem_build_options.h.in

    if [ ! -f $nvshmem_build_options_file ]; then
        echo "missing file at $nvshmem_build_options_file"
        exit_code=1
        return 0;
    fi

    # if the cuda_runtime.h header is not properly included, we fail to parse anything
    # under a __device__ function. This will lead to IWYU falsely omitting headers which
    # may cause failures at compile time.
    while read -r line; do
        if grep -q -e __device__ $line; then
            if ! grep -q -e cuda_runtime\\.h $line; then
                echo $line >> $2
            fi
        fi

        if grep -q -e CUDA_VERSION $line; then
            if ! grep -q -e cuda\\.h $line; then
                echo $line >> $3
            fi
        fi

        if grep -q -e NVML_API_VERSION $line; then
            if ! grep -q -e nvml\\.h $line; then
                echo $line >> $4
            fi
        fi

        if [[ $line == *"nvshmem_build_options.h"* ]]; then
            continue;
        fi

        if [[ $line == *"nvshmem_version.h"* ]]; then
            continue;
        fi

        for opt in $(grep -e cmakedefine $nvshmem_build_options_file | cut -d ' ' -f 2 | xargs); do
            if grep -w -q -e $opt $line; then
                if ! grep -q -e nvshmem_build_options.h $line; then
                    echo "$line missing $opt" >> $5
                    break;
                fi
            fi
        done
    done < $1
}

do_tests() {
    final_list_file=$script_tld/nvshmem_files_final.txt
    missing_cuda_runtime_file=$script_tld/missing_cuda_runtime.txt
    missing_cuda_file=$script_tld/missing_cuda.txt
    missing_nvml_file=$script_tld/missing_nvml.txt
    missing_options_file=$script_tld/missing_nvshmem_build_options.txt

    iwyu_output_file=$script_tld/nvshmem_headers_iwyu_output.txt
    iwyu_output_compilation_errors_file=$script_tld/iwyu_errors.txt
    iwyu_parsed_output_file=$script_tld/iwyu_parsed_output.txt

    touch $final_list_file
    touch $missing_cuda_runtime_file
    touch $missing_cuda_file
    touch $missing_nvml_file
    touch $missing_options_file

    touch $iwyu_output_file
    touch $iwyu_output_compilation_errors_file
    touch $iwyu_parsed_output_file

    cat $missing_cuda_runtime_file >> $1
    cat $1 | sort | uniq -u > $final_list_file

    check_iwyu_quirks $1 $missing_cuda_runtime_file $missing_cuda_file $missing_nvml_file $missing_options_file

    while read -r line; do
        $2 $line 2>&1 || true
    done < $1 >> $iwyu_output_file

    cat $iwyu_output_file | grep ": error:" | grep nvshmem | cut -d ':' -f 1,4,5 \
    | sort | uniq >> $iwyu_output_compilation_errors_file || true
     $nvshmem_home/scripts/iwyu_output_parser.py $iwyu_output_file >> $iwyu_parsed_output_file

    detect_failures $missing_cuda_runtime_file "$3 not including cuda_runtime.h:" || exit_code=1
    detect_failures $missing_cuda_file "$3 not including cuda.h:" || exit_code=1
    detect_failures $missing_nvml_file "$3 not including nvml.h:" || exit_code=1
    detect_failures $missing_options_file "$3 not including nvshmemi_build_options.h:" || exit_code=1
    detect_failures $iwyu_output_compilation_errors_file "compiler errors from the iwyu $3 check" || exit_code=1
    detect_failures $iwyu_parsed_output_file "mismatched dependencies from the iwyu $3 check" || exit_code=1

    rm $missing_cuda_runtime_file $missing_cuda_file $missing_nvml_file $missing_options_file
    rm $final_list_file $iwyu_output_file $iwyu_output_compilation_errors_file $iwyu_parsed_output_file

    return 0
}

do_c_cpp_tests() {
    missing_cuda_runtime_file=$script_tld/missing_cuda_runtime.txt
    missing_cuda_file=$script_tld/missing_cuda.txt
    missing_nvml_file=$script_tld/missing_nvml.txt
    missing_options_file=$script_tld/missing_nvshmem_build_options.txt

    touch $missing_cuda_runtime_file
    touch $missing_cuda_file
    touch $missing_nvml_file
    touch $missing_options_file

    check_iwyu_quirks $1 $missing_cuda_runtime_file $missing_cuda_file $missing_nvml_file $missing_options_file

    detect_failures $missing_cuda_runtime_file "$3 not including cuda_runtime.h:" || exit_code=1
    detect_failures $missing_cuda_file "$3 not including cuda.h:" || exit_code=1
    detect_failures $missing_nvml_file "$3 not including nvml.h:" || exit_code=1
    detect_failures $missing_options_file "$3 not including nvshmemi_build_options.h:" || exit_code=1

    rm $missing_cuda_runtime_file $missing_cuda_file $missing_nvml_file $missing_options_file
}

mkdir -p $script_tld
touch $nvshmem_header_file
touch $nvshmem_cuda_file
touch $nvshmem_c_cpp_file

find $nvshmem_home/src/* -depth -name *.cuh -o -name *.h -o -name *.hpp \
| sort | uniq >> $nvshmem_header_file

find $nvshmem_home/src/* -depth -name *.cu \
| sort | uniq >> $nvshmem_cuda_file

find $nvshmem_home/src/* -depth -name *.c -o -name *.cpp \
| sort | uniq >> $nvshmem_c_cpp_file

cat $nvshmem_header_file | xargs dirname \
| sort | uniq > $nvshmem_header_directories_file

iwyu_command=$iwyu_command_prefix
while read -r line; do
    iwyu_command="$iwyu_command -Xiwyu --check_also=$line/*.h"
    iwyu_command="$iwyu_command -Xiwyu --check_also=$line/*.cuh"
    iwyu_command="$iwyu_command -Xiwyu --check_also=$line/*.hpp -I $line";
done < $nvshmem_header_directories_file
iwyu_command="$iwyu_command $iwyu_command_suffix"

do_c_cpp_tests $nvshmem_c_cpp_file
do_tests $nvshmem_header_file "$iwyu_command" "header"
do_tests $nvshmem_cuda_file "$iwyu_command" "cuda"

if [ "$exit_code" -eq "0" ]; then
    echo -e $formatting_string
    echo "No $4 include issues found"
    echo -e $formatting_string
fi

rm -r $script_tld || true

exit $exit_code
