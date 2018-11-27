#include "NoGui.h"

#include <iostream>

#include <SFML/System.hpp>

#include "Utils.h"

namespace
{
    void printGrid(const std::vector<unsigned char>& grid, size_t width)
    {
        const size_t height = grid.size() / width;
        for (size_t y = 0; y < height; ++y)
        {
            for (size_t x = 0; x < width; ++x)
            {
                std::cout << ((grid[y * width + x]) ? 1 : 0);
                if (x < width - 1)
                    std::cout << " ";
            }
            std::cout << "\n";
        }
    }
}

extern void updateGridOnGPU(unsigned char* grid, size_t gridWidth, size_t gridHeight, size_t blockWidth, size_t blockHeight);

void runIterations(size_t gridWidth, size_t gridHeight, size_t blockWidth, size_t blockHeight, size_t numPasses, bool print, bool test)
{
    if (test)
    {
        gridWidth = (gridWidth < 10) ? 10 : gridWidth;
        gridHeight = (gridHeight < 10) ? 10 : gridHeight;
    }
    std::vector<unsigned char> grid;
    randomizeGrid(grid, gridWidth, gridHeight);
    if (test)
    {
        clearGrid(grid);
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
    }

    if (print)
    {
        std::cout << "Start:\n";
        printGrid(grid, gridWidth);
    }

#ifndef UPDATE_ON_CPU
#define UPDATE_ON_CPU 1
#endif

    sf::Int32 totalTime = 0;
    size_t pass = 0;
    bool runAgain = true;
    while (runAgain)
    {
        sf::Clock clock;
#if UPDATE_ON_CPU == 1
        updateGridOnCPU(grid, gridWidth, gridHeight);
#else
        updateGridOnGPU(&grid.front(), gridWidth, gridHeight, blockWidth, blockHeight);
#endif
        const sf::Time elapsed = clock.getElapsedTime();
        totalTime += elapsed.asMilliseconds();

        if (print)
        {
            std::cout << "Iteration: " << pass + 1 << "\n";
            printGrid(grid, gridWidth);
        }
        ++pass;
        if (numPasses > 0)
        {
            runAgain = (pass < numPasses);
        }
    }
    std::cout << "AVERAGE UPDATE TIME: " << totalTime / (double)pass << "ms for " << pass << " iterations.\n";
}