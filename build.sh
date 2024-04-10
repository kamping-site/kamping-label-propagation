#!/bin/bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target KaMPIngLabelPropagation --parallel
