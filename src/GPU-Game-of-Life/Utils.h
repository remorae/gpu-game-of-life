#pragma once

#include <vector>
#include <cstddef>

void randomizeGrid(std::vector<unsigned char>& grid, size_t width, size_t height);
void clearGrid(std::vector<unsigned char>& grid);
void updateGridOnCPU(std::vector<unsigned char>& grid, size_t width, size_t height);
void setupTest(std::vector<unsigned char>& grid, size_t gridWidth, int test);
void resizeGridForTest(size_t& gridWidth, size_t& gridHeight, int test);