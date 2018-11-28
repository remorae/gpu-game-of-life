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

void runIterations(size_t gridWidth, size_t gridHeight, size_t blockWidth, size_t blockHeight, size_t numPasses, bool print, int test)
{
    std::vector<unsigned char> grid;
    if (test > 0)
    {
        std::cout << "Running test #" << test << ".\n";
        resizeGridForTest(gridWidth, gridHeight, test);
        if (blockWidth >= gridWidth)
            blockWidth = gridWidth / 2;
        if (blockHeight >= gridHeight)
            blockHeight = gridHeight / 2;
        if (blockWidth < 1)
            blockWidth = 1;
        if (blockHeight < 1)
            blockHeight = 1;
        if (blockWidth == 1 && blockHeight == 1)
        {
            blockWidth = 2;
        }
    }

    // Initialize grid
    randomizeGrid(grid, gridWidth, gridHeight);
    
    if (test > 0)
    {
        setupTest(grid, gridWidth, test);
    }

    if (print)
    {
        std::cout << "Start:\n";
        printGrid(grid, gridWidth);
    }

#ifndef UPDATE_ON_CPU
#define UPDATE_ON_CPU 1
#endif

    sf::Time totalTime;
    size_t pass;
    sf::Clock clock;
    for (pass = 0; pass < numPasses; ++pass)
    {
        const std::vector<unsigned char> previous(grid);
        clock.restart();
#if UPDATE_ON_CPU == 1
        updateGridOnCPU(grid, gridWidth, gridHeight);
#else
        updateGridOnGPU(&grid.front(), gridWidth, gridHeight, blockWidth, blockHeight);
#endif
        if (pass != 0)
            totalTime += clock.getElapsedTime();

        if (grid == previous)
        {
            std::cout << "Stopping early due to stable grid state.\n";
            ++pass;
            break;
        }

        if (print && pass < numPasses - 1)
        {
            std::cout << "Iteration: " << pass + 1 << "\n";
            printGrid(grid, gridWidth);
        }
    }
    std::cout << "Final state after " << pass << " passes:\n";
    printGrid(grid, gridWidth);
    std::cout << "AVERAGE UPDATE TIME: " << totalTime.asMilliseconds() / (float)(pass - 1) << "ms for " << pass - 1 << " iterations.\n";
}