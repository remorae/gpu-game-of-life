#pragma once

#include <SFML/Graphics.hpp>

class Game;

struct GraphicsConfig
{
    bool drawLines = false;
    float gridViewportHeight = 1.0f;
    float gridElementWidth = 5.0f;
    float zoomFactor = 0.10f;
};

class GraphicsHandler
{
private:
    GraphicsConfig options;
    sf::View gridView;
    float zoomLevel = 1.0f;

public:
    GraphicsHandler(GraphicsConfig options);

    void init(const Game& game);
    void draw(sf::RenderWindow& window, const Game& game) const;
    void handleZoom(float delta);
    void handlePan(const sf::Vector2f& delta);

    float getCellWidth() const { return options.gridElementWidth; }

private:
    void drawGrid(sf::RenderWindow& window, const Game& game) const;
    void drawUI(sf::RenderWindow& window, const Game& game) const;
};

namespace DrawUtils
{
    void drawLine(sf::RenderWindow& window, const sf::Vector2f& start, const sf::Vector2f& end);
}