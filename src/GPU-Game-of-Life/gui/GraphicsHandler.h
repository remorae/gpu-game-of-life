#pragma once

#include <SFML/Graphics.hpp>

class Game;

struct GraphicsConfig
{
    bool drawLines{ false };
    float gridViewportHeight{ 1.0f };
    float cellWidth{ 5.0f };
    float lineWidth{ 0.0f };
    float zoomFactor{ 0.10f };
    sf::Color backgroundColor{ 50, 50, 50 };
    sf::Color aliveCellColor{ sf::Color::White };
    sf::Color deadCellColor{ sf::Color::Black };
    sf::Color cellOverlayColor{ 255, 0, 0, 150 };
};

class GraphicsHandler
{
private:
    GraphicsConfig options;
    sf::View gridView;
    float zoomLevel{ 1.0f };

public:
    GraphicsHandler(GraphicsConfig options, const sf::Vector2<size_t>& gridSize);

    void draw(sf::RenderWindow& window, const Game& game) const;

    void handleZoom(float delta);
    void handlePan(const sf::Vector2f& delta);
    void resetCamera(const sf::Vector2<size_t>& gridSize);

    float getCellWidth() const { return options.cellWidth; }
    sf::Vector2f pixelToGridCoordinates(const sf::RenderWindow& window, const sf::Vector2i& pixel) const;

private:
    void drawGrid(sf::RenderWindow& window, const Game& game) const;
    void drawUI(sf::RenderWindow& window, const Game& game) const;
};

namespace DrawUtils
{
    void drawLine(sf::RenderWindow& window, const sf::Vector2f& start, const sf::Vector2f& end);
}