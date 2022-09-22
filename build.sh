target="./program"
if ! [[ -f "$target" ]]; then
    rm ./program
fi
clang -O3 -Wall -march=native -lstdc++ main.cpp -o program