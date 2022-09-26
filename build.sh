#!/bin/bash
compile_cpp() {
    target="./program_cpp"
    if ! [[ -f "$target" ]]; then
        rm ./cpp_matmul
    fi
    clang -O3 -Wall -march=native -lstdc++ main.cpp -o cpp_matmul
}

compile_c() {
    target="./program_c"
    if ! [[ -f "$target" ]]; then
        rm c_matmul
    fi
    clang -O3 -Wall -march=native cimpl.c -o c_matmul
}

case $1 in 
    cpp)
    compile_cpp
    ;;
    c)
    compile_c
    ;;
    *)
    compile_c
    compile_cpp
esac