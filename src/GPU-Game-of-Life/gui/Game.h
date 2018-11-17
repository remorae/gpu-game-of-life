#pragma once

#include <array>
#include <string>

#include <SFML/Graphics.hpp>

#include "Button.h"
#include "GraphicsHandler.h"
#include "InputHandler.h"

struct GameConfig
{
    unsigned int screenWidth = 600;
    unsigned int screenHeight = 600;
    std::string title = "GPU Game of Life";
    size_t targetUPS = 8;
    size_t gridWidth = 100;
    size_t gridHeight = 100;
};

class Game
{
private:
    sf::RenderWindow window;
    
    const GameConfig options;
    
    const sf::Time updateThreshold;

    static constexpr size_t kDeltas = 100;
    std::array<float, kDeltas> frameDeltas;
    std::array<float, kDeltas> updateDeltas;

    GraphicsHandler graphicsHandler;
    InputHandler inputHandler;
    std::vector<Button> buttons;

    std::vector<unsigned char> grid;

    bool paused = false;

public:
    Game(GameConfig gameOptions, GraphicsConfig graphicsOptions);

    void run();
    void initGUI();
    void update(const sf::Time& elapsed);
    void draw();

    unsigned char getCell(size_t column, size_t row) const;
    void setCell(size_t column, size_t row, unsigned char value);
    size_t getWidth() const { return options.gridWidth; }
    size_t getHeight() const { return options.gridHeight; }
    const std::vector<Button>& getButtons() const { return buttons; }
};