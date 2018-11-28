#include "Game.h"

#include <iostream>
#include <numeric>

#include "GraphicsHandler.h"
#include "InputHandler.h"
#include "../Utils.h"

Game::Game(GameConfig gameOptions, GraphicsConfig graphicsOptions) :
    window{ sf::VideoMode{ gameOptions.screenWidth, gameOptions.screenHeight }, gameOptions.title },
    options{ std::move(gameOptions) },
    updateThreshold{ sf::seconds(1.0f / gameOptions.targetUPS) },
    graphics{ std::move(graphicsOptions), gridDimensions() },
    input{ InputConfig{} }
{
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

namespace
{
    void repositionButtons(std::vector<Button>& buttons, const GameConfig& options)
    {
        for (size_t i = 1; i < buttons.size(); ++i)
        {
            const sf::FloatRect& previousBounds = buttons[i - 1].getGlobalBounds();
            buttons[i].setPosition(previousBounds.left + previousBounds.width + options.buttonSpacing, previousBounds.top);
        }
    }
}

void Game::togglePaused()
{
    paused = !paused;
    Button& pauseBtn = buttons[2];
    const sf::Text* pauseText = pauseBtn.getText();
    const sf::Font* pauseFont = pauseBtn.getFont();
    if (pauseText && pauseFont)
    {
        sf::Text newText{ (paused) ? "Unpause" : "Pause", *pauseFont, pauseText->getCharacterSize() };
        pauseBtn.setText(newText, *pauseFont);
        repositionButtons(buttons, options);
    }
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
    sf::RectangleShape btnRect{ sf::Vector2f{ 0.0f, options.buttonHeight } }; // Width is determined by text
    btnRect.setFillColor(options.buttonBackgroundColor);

    Button randomBtn{ btnRect, [this]() { randomizeGrid(grid, options.gridWidth, options.gridHeight); } };
    Button clearBtn{ btnRect, [this]() { clearGrid(grid); } };
    Button pauseBtn{ btnRect, [this]() { togglePaused(); } };
    Button resetBtn{ btnRect, [this]() { graphics.resetCamera(gridDimensions()); } };

    sf::Font font;
    if (!font.loadFromFile("resources/fonts/FreeSans.ttf"))
        return;

    sf::Text text;
    text.setCharacterSize(options.buttonCharacterSize);
    text.setFillColor(options.buttonTextColor);

    text.setString("Random");
    randomBtn.setText(text, font);
    const float yPosition = options.screenHeight - randomBtn.getGlobalBounds().height;
    randomBtn.setPosition(0.0f, yPosition);
    buttons.emplace_back(std::move(randomBtn));

    text.setString("Clear");
    clearBtn.setText(text, font);
    buttons.emplace_back(std::move(clearBtn));

    text.setString("Pause");
    pauseBtn.setText(text, font);
    buttons.emplace_back(std::move(pauseBtn));

    text.setString("Reset Camera");
    resetBtn.setText(text, font);
    buttons.emplace_back(std::move(resetBtn));

    repositionButtons(buttons, options);
}

void Game::run()
{
    if (options.runTest > 0)
    {
        resizeGridForTest(options.gridWidth, options.gridHeight, options.runTest);
    }

    randomizeGrid(grid, options.gridWidth, options.gridHeight);

    if (options.runTest > 0)
    {
        setupTest(grid, options.gridWidth, options.runTest);
    }

    initGUI();

    sf::Clock clock;

    sf::Time accumulated;
    size_t frameIndex = 0;
    size_t updateIndex = 0;
    bool incompleteUpdateAverage = true;
    while (window.isOpen())
    {
        const sf::Time elapsed = clock.restart();
        if (!paused)
            accumulated += elapsed;
        
        frameDeltas[frameIndex] = elapsed.asSeconds();
        ++frameIndex;
        frameIndex %= kDeltas;

        if (accumulated >= updateThreshold)
        {
            sf::Clock updateClock;

            update(updateThreshold);
            
            updateDeltas[updateIndex] = updateClock.getElapsedTime().asMilliseconds();
            ++updateIndex;
            if (updateIndex == kDeltas - 1)
                incompleteUpdateAverage = false;
            updateIndex %= kDeltas;

            accumulated -= updateThreshold;
        }

        if (frameIndex == kDeltas - 1)
        {
            if (incompleteUpdateAverage)
                std::cout << "<" << kDeltas << " updates recorded.\n";
            const float averageUpdate = static_cast<float>(std::accumulate(updateDeltas.begin(), updateDeltas.end(), 0)) / ((incompleteUpdateAverage) ? updateIndex : kDeltas);
            const float averageDraw = std::accumulate(frameDeltas.begin(), frameDeltas.end(), 0.0f) / kDeltas;
            std::cout << "Avg. Update Time: " << averageUpdate << "ms, FPS: " << 1.0f / averageDraw << "\n";
        }

        draw();
        input.handleEvents(window, *this, graphics);
    }
}
extern void updateGridOnGPU(unsigned char* grid, size_t gridWidth, size_t gridHeight, size_t blockWidth, size_t blockHeight);

void Game::update(const sf::Time& elapsed)
{
#ifndef UPDATE_ON_CPU
#define UPDATE_ON_CPU 1
#endif

#if UPDATE_ON_CPU == 1
    updateGridOnCPU(grid, options.gridWidth, options.gridHeight);
#else
    updateGridOnGPU(&grid.front(), options.gridWidth, options.gridHeight, options.blockWidth, options.blockHeight);
#endif
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

void Game::pan(const sf::Vector2i& direction)
{
    graphics.handlePan({ static_cast<float>(direction.x) * options.arrowPanFactor, static_cast<float>(direction.y) * options.arrowPanFactor });
}
