#include "Game.h"

#include <iostream>
#include <numeric>
#include <random>

#include "GraphicsHandler.h"
#include "InputHandler.h"

Game::Game(GameConfig gameOptions, GraphicsConfig graphicsOptions) :
    window{ sf::VideoMode{ gameOptions.screenWidth, gameOptions.screenHeight }, gameOptions.title },
    options{ std::move(gameOptions) },
    updateThreshold{ sf::seconds(1.0f / gameOptions.targetUPS) },
    graphicsHandler{ std::move(graphicsOptions), getWidth(), getHeight() },
    inputHandler{ InputConfig{} }
{
}

namespace
{
    void randomizeGrid(std::vector<unsigned char>& grid, size_t width, size_t height)
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        static std::uniform_int_distribution<unsigned int> distribution{0, 1};

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
        for (int i = 0; i < grid.size(); ++i)
            grid[i] = 0;
    }
}

sf::Vector2f Game::pixelToGridCoordinates() const
{
    return graphicsHandler.pixelToGridCoordinates(window);
}

void Game::initGUI()
{
    sf::RectangleShape btnRect{ sf::Vector2f{ 95.0f, 30.0f } };
    btnRect.setFillColor(sf::Color::Blue);
    Button randomBtn{ btnRect, [this]() { randomizeGrid(grid, options.gridWidth, options.gridHeight); } };

    btnRect.setSize({ 65.0f, 30.0f });
    Button clearBtn{ btnRect, [this]() { clearGrid(grid); } };

    btnRect.setSize({ 70.0f, 30.0f });
    Button pauseBtn{ btnRect, [this]() { paused = !paused; } };

    btnRect.setSize({ 160.0f, 30.0f });
    Button resetBtn{ btnRect, [this]() { graphicsHandler.resetCamera(getWidth(), getHeight()); } };
    
    sf::Font font;
    if (font.loadFromFile("resources/fonts/FreeSans.ttf"))
    {
        sf::Text text;
        text.setString("Random");
        text.setCharacterSize(24);
        text.setFillColor(sf::Color::White);
        randomBtn.setText(text, font);
        randomBtn.setPosition(0.0f, 570.0f);
        buttons.emplace_back(std::move(randomBtn));

        text.setString("Clear");
        clearBtn.setText(text, font);
        clearBtn.setPosition(100.0f, 570.0f);
        buttons.emplace_back(std::move(clearBtn));

        text.setString("Pause");
        pauseBtn.setText(text, font);
        pauseBtn.setPosition(170.0f, 570.0f);
        buttons.emplace_back(std::move(pauseBtn));

        text.setString("Reset Camera");
        resetBtn.setText(text, font);
        resetBtn.setPosition(245.0f, 570.0f);
        buttons.emplace_back(std::move(resetBtn));
    }
}

void Game::run()
{
    randomizeGrid(grid, options.gridWidth, options.gridHeight);
    initGUI();

    sf::Clock clock;

    sf::Time accumulated;
    size_t frameIndex = 0;
    size_t updateIndex = 0;
    while (window.isOpen())
    {
        const sf::Time elapsed = clock.restart();
        if (!paused)
            accumulated += elapsed;
        
        frameDeltas[frameIndex] = elapsed.asSeconds();
        frameIndex = ++frameIndex % kDeltas;

        if (accumulated >= updateThreshold)
        {
            updateDeltas[updateIndex] = accumulated.asSeconds();
            updateIndex = ++updateIndex % kDeltas;

            update(updateThreshold);
            accumulated -= updateThreshold;
        }

        if (frameIndex == kDeltas - 1)
        {
            const float averageUpdate = std::accumulate(updateDeltas.begin(), updateDeltas.end(), 0.0f) / kDeltas;
            const float averageDraw = std::accumulate(frameDeltas.begin(), frameDeltas.end(), 0.0f) / kDeltas;
            std::cout << "UPS: " << 1.0f / averageUpdate << ", FPS: " << 1.0f / averageDraw << "\n";
        }

        draw();
        inputHandler.handleEvents(window, *this, graphicsHandler);
    }
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
}

void Game::update(const sf::Time& elapsed)
{
    updateGridOnCPU(grid, options.gridWidth, options.gridHeight);
}

void Game::draw()
{
    window.clear();
    graphicsHandler.draw(window, *this);
    window.display();
}

unsigned char Game::getCell(size_t column, size_t row) const
{
    if (row >= options.gridHeight || column >= options.gridWidth)
        throw std::invalid_argument{ "Invalid element coordinates." };

    const size_t index = row * options.gridWidth + column;
    if (grid.size() < index)
        throw std::logic_error{ "Bad grid state; too small for current grid dimensions." };

    return grid[index];
}

void Game::setCell(size_t column, size_t row, unsigned char value)
{
    if (row >= options.gridHeight || column >= options.gridWidth)
        throw std::invalid_argument{ "Invalid element coordinates." };

    const size_t index = row * options.gridWidth + column;
    if (grid.size() < index)
        throw std::logic_error{ "Bad grid state; too small for current grid dimensions." };

    grid[index] = value;
}
