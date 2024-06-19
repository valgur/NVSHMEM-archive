#! /bin/bash

# check for includes breaking one of the four ABI barriers.
# Takes one argument (NVSHMEM Top Level Directory)
# returns 0 if ABI is preserved, otherwise returns 1
# Prints all ABI breaking files and the breaking includes

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_nvshmem>"
    exit 1
fi

fail=0

NVSHMEM_INCLUDE_PATH=$1/src/include
NVSHMEM_INTERNAL_INCLUDE_PATH=$1/src/include/internal
NVSHMEM_SOURCE_PATH=$1/src
NVSHMEM_MODULE_PATH=$1/src/modules

#$1 top level directory array
#$2 directories to check against
#$3 top level source directory
#$4 top level include directory
check_includes() {
    local -n tlds=$1
    local -n dirs_to_check=$2
    include_prefix=$4

    for item in ${tlds[@]}; do
        directory=`basename $item`
        for file in `find $item -type f`; do
            for item2 in ${dirs_to_check[@]}; do
                check=`basename $item2`
                check=${include_prefix}${check}
                if [[ ! $check =~ $directory ]]; then
                    grep -q "#include \"$check/" $file && echo -e "$file includes $item2 headers" && fail=1
                    grep "#include \"$check/" $file
                fi
            done
        done
    done
}

internal_include_tlds=(`find $NVSHMEM_INTERNAL_INCLUDE_PATH -mindepth 1 -maxdepth 1 -path $NVSHMEM_INTERNAL_INCLUDE_PATH/non_abi -prune -o -type d -print`)
include_tlds=(`find $NVSHMEM_INCLUDE_PATH -mindepth 1 -maxdepth 1 -path $NVSHMEM_INCLUDE_PATH/non_abi -prune -o -path $NVSHMEM_INCLUDE_PATH/internal -prune -o -type d -print`)
module_tlds=(`find $NVSHMEM_MODULE_PATH -mindepth 1 -maxdepth 1 -type d`)
source_tlds=(`find $NVSHMEM_SOURCE_PATH -mindepth 1 -maxdepth 1 -path $NVSHMEM_SOURCE_PATH/include -o -prune -path $NVSHMEM_SOURCE_PATH/modules -prune -o -prune -path $NVSHMEM_SOURCE_PATH/bin -prune -o -type d -print`)

include_dirs_to_check=(`find $NVSHMEM_INCLUDE_PATH -mindepth 1 -maxdepth 1 -type d -path $NVSHMEM_INCLUDE_PATH/non_abi -o -type d -print`)

echo "printing ABI incompatible includes, if any"

check_includes include_tlds include_dirs_to_check $NVSHMEM_INCLUDE_PATH ""
check_includes internal_include_tlds include_tlds $NVSHMEM_INCLUDE_PATH ""

check_includes source_tlds include_tlds $NVSHMEM_INCLUDE_PATH ""
check_includes source_tlds internal_include_tlds  $NVSHMEM_INTERNAL_INCLUDE_PATH "internal/"

check_includes module_tlds include_tlds $NVSHMEM_INCLUDE_PATH ""
check_includes module_tlds internal_include_tlds  $NVSHMEM_INTERNAL_INCLUDE_PATH "internal/"

exit $fail
