#pragma once

#include <array>
#include <memory>
#include <string>

#include <SFML/Graphics.hpp>

#include "Button.h"
#include "GraphicsHandler.h"
#include "InputHandler.h"

struct GameConfig
{
    unsigned int screenWidth{ 600 };
    unsigned int screenHeight{ 600 };
    std::string title{ "GPU Game of Life" };
    size_t targetUPS{ 8 };
    size_t gridWidth{ 100 };
    size_t gridHeight{ 100 };
    size_t blockWidth{ 32 };
    size_t blockHeight{ 32 };
    float buttonHeight{ 30.0f };
    float buttonSpacing{ 5.0f };
    sf::Color buttonTextColor{ sf::Color::White };
    sf::Color buttonBackgroundColor{ sf::Color::Blue };
    unsigned int buttonCharacterSize{ 24 };
    int runTest{ 0 };
    float arrowPanFactor{ 5.0f };
};

class Game
{
private:
    sf::RenderWindow window;

    GameConfig options;

    const sf::Time updateThreshold;

    static constexpr size_t kDeltas{ 100 };
    std::array<float, kDeltas> frameDeltas;
    std::array<sf::Int32, kDeltas> updateDeltas;

    GraphicsHandler graphics;
    InputHandler input;
    std::vector<Button> buttons;

    std::vector<unsigned char> grid;
    CellEditMode editMode;

    bool paused = false;

public:
    Game(GameConfig gameOptions, GraphicsConfig graphicsOptions);

    void run();

    unsigned char getCell(const sf::Vector2<size_t>& location) const;
    void editCell(const sf::Vector2<size_t>& location);
    void setEditModeFromLocation(const sf::Vector2<size_t>& location);
    std::unique_ptr<sf::Vector2<size_t>> getOverlayCoordinates() const;

    sf::Vector2<size_t> gridDimensions() const { return { options.gridWidth, options.gridHeight }; }

    const std::vector<Button>& getButtons() const { return buttons; }

    void handleMouseMove(bool panning, const sf::Vector2f& distance);
    void pan(const sf::Vector2i& direction);
    void handlePrimaryClick(const sf::Event& event);

private:
    void initGUI();
    void update(const sf::Time& elapsed);
    void draw();

    sf::Vector2f getMousePositionOnGrid() const;
    std::unique_ptr<sf::Vector2<size_t>> cellCoordinatesFromPosition(const sf::Vector2f& position) const;
    bool isPositionWithinGrid(const sf::Vector2<size_t>& position) const;
    void togglePaused();
};