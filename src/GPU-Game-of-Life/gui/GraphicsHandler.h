#pragma once

#include <SFML/Graphics.hpp>

class Game;

struct GraphicsConfig
{
    bool drawLines{ false };
    float gridViewportHeight{ 1.0f };
    float gridElementWidth{ 5.0f };
    float zoomFactor{ 0.10f };
    sf::Color backgroundColor{ 50, 50, 50 };
    sf::Color aliveCellColor{ sf::Color::White };
    sf::Color deadCellColor{ sf::Color::Black };
};

class GraphicsHandler
{
private:
    GraphicsConfig options;
    sf::View gridView;
    float zoomLevel{ 1.0f };

public:
    GraphicsHandler(GraphicsConfig options, size_t gridWidth, size_t gridHeight);

    void draw(sf::RenderWindow& window, const Game& game) const;

    void handleZoom(float delta);
    void handlePan(const sf::Vector2f& delta);
    void resetCamera(size_t gridWidth, size_t gridHeight);

    float getCellWidth() const { return options.gridElementWidth; }
    sf::Vector2f pixelToGridCoordinates(const sf::RenderWindow& window) const;

private:
    void drawGrid(sf::RenderWindow& window, const Game& game) const;
    void drawUI(sf::RenderWindow& window, const Game& game) const;
};

namespace DrawUtils
{
    void drawLine(sf::RenderWindow& window, const sf::Vector2f& start, const sf::Vector2f& end);
}