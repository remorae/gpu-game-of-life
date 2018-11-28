#include <iostream>
#include <string>

#include <SFML/Graphics.hpp>

#include "gui/Game.h"
#include "gui/GraphicsHandler.h"
#include "NoGui.h"

namespace
{
    void parsePrintFlag(const std::string& arg, bool& print)
    {
        if (arg == "p")
        {
            print = true;
        }
        else
        {
            throw std::invalid_argument("Unknown parameter.");
        }
    }

    void parsePasses(const std::string& arg, size_t& numPasses)
    {
        numPasses = std::stoul(arg, nullptr, 10);
        if (numPasses == 0)
            throw std::invalid_argument("numPasses must be positive.");
    }

    void parseTest(const std::string& arg, int& test)
    {
        if (arg.length() != 2)
            throw std::invalid_argument("Unknown parameter.");

        switch (arg[1])
        {
            case '1':
                test = 1;
                break;
            case '2':
                test = 2;
                break;
            default:
                throw std::invalid_argument("Unknown parameter.");
        }
    }

    void parseOptionalParam(const std::string& arg, bool& print, size_t& numPasses, int& test)
    {
        try
        {
            parsePrintFlag(arg, print);
        }
        catch (const std::exception&)
        {
            try
            {
                parsePasses(arg, numPasses);
            }
            catch (const std::exception&)
            {
                try
                {
                    parseTest(arg, test);
                }
                catch (const std::exception&)
                {
                    throw std::invalid_argument("Unknown parameter.");
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    bool gui = true;
    bool print = false;
    int test = 0;
    size_t numPasses = 0;

    size_t blockWidth = 32;
    size_t blockHeight = 32;
    size_t gridHeight = 100;
    size_t gridWidth = 100;

    try
    {
        if (argc < 5)
        {
            std::string msg("Invalid arguments: blockWidth blockHeight gridWidth gridHeight [numPasses] [p] [t<1|2>]\n");
            msg = msg + "numPasses: Set to a positive integer to disable the GUI and update the grid the given # of times. e.g. 100\n";
            msg = msg + "p: Include to print the grid every iteration when the GUI is disabled.\n";
            msg = msg + "t#: Include to create a simple test case rather than a random grid at startup. e.g. t2";
            throw std::invalid_argument(msg);
        }

        blockWidth = std::stoul(argv[1], nullptr, 10);
        blockHeight = std::stoul(argv[2], nullptr, 10);
        gridWidth = std::stoul(argv[3], nullptr, 10);
        gridHeight = std::stoul(argv[4], nullptr, 10);

        if (blockWidth * blockHeight > 1024)
        {
            throw std::invalid_argument("blockWidth * blockHeight must be <= 1024.");
        }
        if (blockWidth == 0 || blockHeight == 0 || gridWidth == 0 || gridHeight == 0)
        {
            throw std::invalid_argument("First four parameters must be positive.");
        }

        if (argc > 5)
        {
            if (argc > 8)
            {
                throw std::invalid_argument("Too many parameters.");
            }
            for (int i = 5; i < argc; ++i)
            {
                parseOptionalParam(argv[i], print, numPasses, test);
                if (numPasses > 0)
                    gui = false;
            }
        }
    }
    catch (const std::exception& ex)
    {
        std::cout << ex.what() << "\n";
        std::cout << "Press ENTER...\n";
        std::cin.get();
        return 1;
    }

    if (gui)
    {
        GameConfig config;
        config.gridWidth = gridWidth;
        config.gridHeight = gridHeight;
        config.blockWidth = blockWidth;
        config.blockHeight = blockHeight;
        config.runTest = test;
        Game game{ std::move(config), GraphicsConfig{} };
        game.run();
    }
    else
    {
        runIterations(gridWidth, gridHeight, blockWidth, blockHeight, numPasses, print, test);
    }

    std::cout << "Press ENTER...\n";
    std::cin.get();
    return 0;
}