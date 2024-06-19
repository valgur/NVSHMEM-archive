#! /usr/bin/env python3

# Parses an output from an IWYU run as stored in a file.
# It takes one positional argument, the path to the output file.
# Returns 0 if it executes properly.

import sys

in_loop = False
printed_intro = False
current_includes = []
src_file_list = []

if len(sys.argv) != 2:
    print("Usage: python iwyu_output_parser.py <iwyu_output_file>")
    sys.exit(1)

with open(sys.argv[1], "r") as file:
    my_input = file.readlines()
    for line in my_input:
        if not in_loop:
            if "The full include-list" in line:
                src_file = line.split(" ")[-1].strip().strip(":")
                if src_file not in src_file_list:
                    src_file = line.split(" ")[-1].strip().strip(":")
                    src_file_list.append(src_file)
                    if not printed_intro:
                        print("The full include-list for all failing files follows:")
                        printed_intro = True
                    in_loop = True
        else:
            if "---" in line:
                i = 0
                in_loop = False
                if src_file == "":
                    continue
                print(src_file)
                for line in current_includes:
                    print(line.strip())
                current_includes = []
                src_file = ""
            else:
                current_includes.append(line)
