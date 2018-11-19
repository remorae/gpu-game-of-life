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
    graphics{ std::move(graphicsOptions), gridDimensions() },
    input{ InputConfig{} }
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

std::unique_ptr<sf::Vector2<size_t>> Game::cellCoordinatesFromPosition(const sf::Vector2f& position) const
{
    const sf::Vector2f cellPosition = position / graphics.getCellWidth();
    auto result = std::make_unique<sf::Vector2<size_t>>(static_cast<size_t>(cellPosition.x), static_cast<size_t>(cellPosition.y));
    if (isPositionWithinGrid(*result))
    {
        return result;
    }

    return nullptr;
}

bool Game::isPositionWithinGrid(const sf::Vector2<size_t>& position) const
{
    const auto gridSize = gridDimensions();
    return position.x >= 0 && position.x < gridSize.x
        && position.y >= 0 && position.y < gridSize.y;
}

sf::Vector2f Game::getMousePositionOnGrid() const
{
    const sf::Vector2i windowCoordinates = input.getWindowMousePosition(window);
    return graphics.pixelToGridCoordinates(window, windowCoordinates);
}

void Game::handleMouseMove(bool panning, const sf::Vector2f& distance)
{
    if (panning)
    {
        graphics.handlePan(distance);
    }

    if (const auto cellAtMouse = cellCoordinatesFromPosition(getMousePositionOnGrid()))
    {
        if (editMode != CellEditMode::NONE && input.mouseButtonDown(sf::Mouse::Button::Left))
        {
            editCell(*cellAtMouse);
        }
    }
}

namespace
{
    bool checkButtons(const sf::Event event, const std::vector<Button>& buttons)
    {
        bool anyClicked = false;
        for (const Button& btn : buttons)
        {
            const sf::FloatRect& bounds = btn.getGlobalBounds();
            if (event.mouseButton.x >= bounds.left && event.mouseButton.x < bounds.left + bounds.width
                && event.mouseButton.y >= bounds.top && event.mouseButton.y < bounds.top + bounds.height)
            {
                anyClicked = true;
                btn.clicked();
            }
        }
        return anyClicked;
    }
}

void Game::handlePrimaryClick(const sf::Event& event)
{
    const bool buttonClicked = checkButtons(event, buttons);
    if (!buttonClicked)
    {
        if (const auto cellClicked = cellCoordinatesFromPosition(getMousePositionOnGrid()))
        {
            setEditModeFromLocation(*cellClicked);
            editCell(*cellClicked);
        }
    }
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
    Button resetBtn{ btnRect, [this]() { graphics.resetCamera(gridDimensions()); } };
    
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
        input.handleEvents(window, *this, graphics);
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
    graphics.draw(window, *this);
    window.display();
}

unsigned char Game::getCell(const sf::Vector2<size_t>& location) const
{
    if (location.y >= options.gridHeight || location.x >= options.gridWidth)
        throw std::invalid_argument{ "Invalid element coordinates." };

    const size_t index = location.y * options.gridWidth + location.x;
    if (grid.size() < index)
        throw std::logic_error{ "Bad grid state; too small for current grid dimensions." };

    return grid[index];
}

void Game::editCell(const sf::Vector2<size_t>& location)
{
    if (location.y >= options.gridHeight || location.x >= options.gridWidth)
        throw std::invalid_argument{ "Invalid element coordinates." };

    const size_t index = location.y * options.gridWidth + location.x;
    if (grid.size() < index)
        throw std::logic_error{ "Bad grid state; too small for current grid dimensions." };

    grid[index] = (editMode == CellEditMode::ON) ? 1 : 0;
}

void Game::setEditModeFromLocation(const sf::Vector2<size_t>& location)
{
    editMode = (getCell(location) == 1) ? CellEditMode::OFF : CellEditMode::ON;
}

std::unique_ptr<sf::Vector2<size_t>> Game::getOverlayCoordinates() const
{
    return cellCoordinatesFromPosition(getMousePositionOnGrid());
}
