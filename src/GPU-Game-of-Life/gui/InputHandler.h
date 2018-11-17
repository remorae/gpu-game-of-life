#pragma once

#include <SFML/Graphics.hpp>

class Game;
class GraphicsHandler;

enum class CellEditMode
{
    OFF,
    ON,
    NONE
};

struct InputConfig
{
    bool outputMousePosition{ false };
};

class InputHandler
{
private:
    InputConfig options;
    bool isPanning = false;
    sf::Vector2f previousMousePosition;
    sf::Vector2f currentMousePosition;
    CellEditMode editMode = CellEditMode::NONE;
    std::vector<sf::Mouse::Button> mouseJustPressed;
    std::vector<sf::Mouse::Button> currentMouseDown;
    std::vector<sf::Mouse::Button> mouseJustReleased;

public:
    InputHandler(InputConfig options);
    
    void handleEvents(sf::RenderWindow& window, Game& game, GraphicsHandler& graphicsHandler);

private:
    void setPanning(bool value) { isPanning = value; }
    void onMouseMove(const sf::Event& event, Game& game, GraphicsHandler& graphicsHandler);
    void onMousePress(const sf::Event& event, Game& game, const GraphicsHandler& graphicsHandler);
    void onMouseRelease(const sf::Event& event);

    bool mouseButtonDown(const sf::Mouse::Button& button) const;
    bool mouseButtonPressed(const sf::Mouse::Button& button) const;
    bool mouseButtonReleased(const sf::Mouse::Button& button) const;
};