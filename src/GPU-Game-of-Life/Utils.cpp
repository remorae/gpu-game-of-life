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