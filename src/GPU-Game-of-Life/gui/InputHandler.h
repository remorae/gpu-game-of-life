#pragma once

#include <SFML/Graphics.hpp>

class Game;
class GraphicsHandler;

enum class ToggleMode
{
    OFF,
    ON,
    NONE
};

struct InputConfig
{
    bool outputMousePosition = false;
};

class InputHandler
{
private:
    InputConfig options;
    bool isPanning = false;
    sf::Vector2f previousMousePosition;
    sf::Vector2f currentMousePosition;
    ToggleMode toggleMode = ToggleMode::NONE;
    std::vector<sf::Mouse::Button> previousMouseDown;
    std::vector<sf::Mouse::Button> currentMouseDown;

public:
    InputHandler(InputConfig options);
    
    void handleEvents(sf::RenderWindow& window, Game& game, GraphicsHandler& graphicsHandler);

private:
    void setPanning(bool value) { isPanning = value; }
};