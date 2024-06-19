#!/usr/bin/env bash

# This is a smoketest to confirm IWYU is functioning properly.
# It has both a positive and negative test condition.
# It takes no arguments and returns 0 if both tests pass, 1 otherwise.



if ! hash include-what-you-use 2>/dev/null; then
  echo "include-what-you-use not found in PATH"
  exit 1
else
    cat >/tmp/iwyu_test_passing.cpp <<EOL
#include <stddef.h>
int main() {
    size_t my_size;
}
EOL
    cat >/tmp/iwyu_test_failing.cpp <<EOL
#include <stddef.h>
#include <stdint.h>
int main() {
    size_t my_size;
}
EOL

    iwyu_llvm_version=$(ldd `which iwyu` | grep --only-matching -e 'libLLVM-[[:digit:]]*\.so\.[[:digit:]]' | cut -d '-' -f 2 | cut -d '.' -f 1 | uniq)

    #we can't use return codes here. iwyu returns with non-standard values before v 1.19
    if ! include-what-you-use -std=c++11 /tmp/iwyu_test_passing.cpp 2>&1 | grep -q "iwyu_test_passing.cpp has correct"; then
        echo "include-what-you-use failed to pass on a correct file."
        echo "On ubuntu this likely means libclang-common-${iwyu_llvm_version}-dev is not installed."
        echo "In general, this likely means that /usr/lib/clang/${iwyu_llvm_version}/include/ does not exist."
        exit 1
    fi

    if ! include-what-you-use -std=c++11 /tmp/iwyu_test_failing.cpp 2>&1 | grep -q "iwyu_test_failing.cpp should remove these lines"; then
        echo "include-what-you-use failed to fail on an incorrect file."
        echo "This test is likely broken and needs to be fixed."
        exit 1
    fi

    exit 0
fi