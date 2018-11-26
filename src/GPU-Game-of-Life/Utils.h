#pragma once

#include <vector>

void randomizeGrid(std::vector<unsigned char>& grid, size_t width, size_t height);
void clearGrid(std::vector<unsigned char>& grid);
void updateGridOnCPU(std::vector<unsigned char>& grid, size_t width, size_t height);