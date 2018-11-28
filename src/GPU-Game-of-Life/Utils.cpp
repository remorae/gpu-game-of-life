#include "Utils.h"

#include <random>

void randomizeGrid(std::vector<unsigned char>& grid, size_t width, size_t height)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    static std::uniform_int_distribution<unsigned int> distribution{ 0, 1 };

    const bool resize = (grid.size() != width * height);
    if (resize)
    {
        grid.clear();
        grid.reserve(width * height);
    }
    for (size_t row = 0; row < height; ++row)
    {
        for (size_t col = 0; col < width; ++col)
        {
            const auto value = static_cast<unsigned char>(distribution(generator));
            if (resize)
            {
                grid.push_back(value);
            }
            else
            {
                grid[row * width + col] = value;
            }
        }
    }
}

void clearGrid(std::vector<unsigned char>& grid)
{
    for (size_t i = 0; i < grid.size(); ++i)
        grid[i] = 0;
}

namespace
{
    unsigned int countAliveNeighbors(std::vector<unsigned char>& grid, size_t width, size_t height, size_t row, size_t col)
    {
        unsigned int aliveNeighbors = 0;

        for (int deltaX = -1; deltaX <= 1; ++deltaX)
        {
            for (int deltaY = -1; deltaY <= 1; ++deltaY)
            {
                if (deltaX == 0 && deltaY == 0)
                    continue;
                const size_t y = (height + deltaY + row) % height; // Add height in case of negative
                const size_t x = (width + col + deltaX) % width; // Add width in case of negative
                const auto neighbor = grid[y * width + x];
                if (neighbor)
                    ++aliveNeighbors;
            }
        }

        return aliveNeighbors;
    }
}

void updateGridOnCPU(std::vector<unsigned char>& grid, size_t width, size_t height)
{
    std::vector<unsigned char> result;
    result.reserve(width * height);
    for (size_t i = 0; i < width * height; ++i)
        result.push_back(0);

    for (size_t row = 0; row < height; ++row)
    {
        for (size_t col = 0; col < width; ++col)
        {
            const auto numAliveNeighbors = countAliveNeighbors(grid, width, height, row, col);
            const size_t currentIndex = row * width + col;
            if (grid[currentIndex])
            {
                result[currentIndex] = (numAliveNeighbors == 2 || numAliveNeighbors == 3);
            }
            else
            {
                result[currentIndex] = (numAliveNeighbors == 3);
            }
        }
    }
    grid = result;
}

void setupTest(std::vector<unsigned char>& grid, size_t gridWidth, int test)
{
    clearGrid(grid);

    switch (test)
    {
        case 1:
            // Light-weight spaceship
            grid[1 * gridWidth + 3] = 1;
            grid[1 * gridWidth + 6] = 1;
            grid[2 * gridWidth + 7] = 1;
            grid[3 * gridWidth + 3] = 1;
            grid[3 * gridWidth + 7] = 1;
            grid[4 * gridWidth + 4] = 1;
            grid[4 * gridWidth + 5] = 1;
            grid[4 * gridWidth + 6] = 1;
            grid[4 * gridWidth + 7] = 1;
            break;
        case 2:
            // Gosper glider gun
            grid[15 * gridWidth + 7] = 1;
            grid[16 * gridWidth + 7] = 1;
            grid[15 * gridWidth + 8] = 1;
            grid[16 * gridWidth + 8] = 1;
            grid[12 * gridWidth + 17] = 1;
            grid[13 * gridWidth + 17] = 1;
            grid[17 * gridWidth + 17] = 1;
            grid[18 * gridWidth + 17] = 1;
            grid[13 * gridWidth + 18] = 1;
            grid[14 * gridWidth + 18] = 1;
            grid[15 * gridWidth + 18] = 1;
            grid[16 * gridWidth + 18] = 1;
            grid[17 * gridWidth + 18] = 1;
            grid[13 * gridWidth + 19] = 1;
            grid[14 * gridWidth + 19] = 1;
            grid[16 * gridWidth + 19] = 1;
            grid[17 * gridWidth + 19] = 1;
            grid[13 * gridWidth + 20] = 1;
            grid[14 * gridWidth + 20] = 1;
            grid[16 * gridWidth + 20] = 1;
            grid[17 * gridWidth + 20] = 1;
            grid[14 * gridWidth + 21] = 1;
            grid[15 * gridWidth + 21] = 1;
            grid[16 * gridWidth + 21] = 1;
            grid[17 * gridWidth + 29] = 1;
            grid[16 * gridWidth + 30] = 1;
            grid[17 * gridWidth + 30] = 1;
            grid[18 * gridWidth + 30] = 1;
            grid[15 * gridWidth + 31] = 1;
            grid[16 * gridWidth + 31] = 1;
            grid[17 * gridWidth + 31] = 1;
            grid[18 * gridWidth + 31] = 1;
            grid[19 * gridWidth + 31] = 1;
            grid[14 * gridWidth + 32] = 1;
            grid[16 * gridWidth + 32] = 1;
            grid[18 * gridWidth + 32] = 1;
            grid[20 * gridWidth + 32] = 1;
            grid[14 * gridWidth + 33] = 1;
            grid[15 * gridWidth + 33] = 1;
            grid[19 * gridWidth + 33] = 1;
            grid[20 * gridWidth + 33] = 1;
            grid[16 * gridWidth + 41] = 1;
            grid[17 * gridWidth + 41] = 1;
            grid[16 * gridWidth + 42] = 1;
            grid[17 * gridWidth + 42] = 1;
            break;
        default:
            break;
    }
}

void resizeGridForTest(size_t& gridWidth, size_t& gridHeight, int test)
{
    size_t newGridSize = 1;
    switch (test)
    {
        case 1:
            newGridSize = 10;
            break;
        case 2:
            newGridSize = 45;
            break;
        default:
            break;
    }
    gridWidth = std::max(gridWidth, newGridSize);
    gridHeight = std::max(gridHeight, newGridSize);
}