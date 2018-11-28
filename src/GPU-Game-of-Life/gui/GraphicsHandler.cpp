#include "GraphicsHandler.h"

#include "Button.h"
#include "Game.h"

void DrawUtils::drawLine(sf::RenderWindow& window, const sf::Vector2f& start, const sf::Vector2f& end)
{
    sf::Vertex line[] =
    {
        sf::Vertex{ start },
        sf::Vertex{ end }
    };
    window.draw(line, 2, sf::Lines);
}

GraphicsHandler::GraphicsHandler(GraphicsConfig options, const sf::Vector2<size_t>& gridSize) :
    options(std::move(options))
{
    resetCamera(gridSize);
    gridView.setViewport(sf::FloatRect{ 0.0f, 0.0f, 1.0f, options.gridViewportHeight });
}

namespace
{
    void drawGridLines(sf::RenderWindow& window, const sf::Vector2<size_t>& gridSize,
                       const sf::Vector2f& viewSize, const GraphicsConfig& options)
    {
        // Horizontal
        for (unsigned int y = 0; y <= gridSize.y; ++y)
        {
            const float startY = y * (options.cellWidth + options.lineWidth);
            DrawUtils::drawLine(window, { 0.0f, startY }, { viewSize.x, startY });
        }

        // Vertical
        for (unsigned int x = 0; x <= gridSize.x; ++x)
        {
            const float startX = x * (options.cellWidth + options.lineWidth);
            DrawUtils::drawLine(window, { startX, 0.0f }, { startX, viewSize.y });
        }
    }

    void drawOverlay(sf::RenderWindow& window, const Game& game, const GraphicsConfig& options)
    {
        if (const auto overlayPosition = game.getOverlayCoordinates())
        {
            sf::RectangleShape cell{ sf::Vector2f{ options.cellWidth, options.cellWidth } };
            cell.setFillColor(options.cellOverlayColor);
            const sf::Vector2f position{ overlayPosition->x * options.cellWidth, overlayPosition->y * options.cellWidth };
            cell.setPosition(position);

            window.draw(cell);
        }
    }

    void drawCells(sf::RenderWindow& window, const Game& game, const sf::Vector2<size_t>& gridSize, const GraphicsConfig& options)
    {
        sf::RectangleShape cell{ sf::Vector2f{ options.cellWidth, options.cellWidth } };
        for (size_t y = 0; y < gridSize.y; ++y)
        {
            for (size_t x = 0; x < gridSize.x; ++x)
            {
                cell.setFillColor((game.getCell({ x, y })) ? options.aliveCellColor : options.deadCellColor);
                cell.setPosition({ x * (options.cellWidth + options.lineWidth), y * (options.cellWidth + options.lineWidth) });

                window.draw(cell);
            }
        }

        drawOverlay(window, game, options);
    }
}

void GraphicsHandler::drawGrid(sf::RenderWindow& window, const Game& game) const
{
    const sf::Vector2<size_t> gridSize = game.gridDimensions();
        
    const float totalWidth = ((gridSize.x + 1) * options.lineWidth) + (options.cellWidth * gridSize.x);
    const float totalHeight = ((gridSize.y + 1) * options.lineWidth) + (options.cellWidth * gridSize.y);

    sf::View view{ gridView };
    view.zoom(zoomLevel);
    window.setView(view);
    window.clear(options.backgroundColor);
        
    if (options.drawLines)
    {
        drawGridLines(window, gridSize, { totalWidth, totalHeight }, options);
    }

    drawCells(window, game, gridSize, options);
}

void GraphicsHandler::drawUI(sf::RenderWindow& window, const Game& game) const
{
    window.setView(window.getDefaultView());

    for (const Button& button : game.getButtons())
    {
        button.draw(window);
    }
}

void GraphicsHandler::draw(sf::RenderWindow& window, const Game& game) const
{
    drawGrid(window, game);
    drawUI(window, game);
}

void GraphicsHandler::handleZoom(const float delta)
{
    zoomLevel += options.zoomFactor * delta * -1.0f;
    if (zoomLevel < 0)
        zoomLevel = 0.01f;
}

void GraphicsHandler::handlePan(const sf::Vector2f& delta)
{
    gridView.move(delta.x, delta.y);
}

void GraphicsHandler::resetCamera(const sf::Vector2<size_t>& gridSize)
{
    // Make sure the view is always square to prevent stretching/squishing
    size_t maxSide = std::max(gridSize.x, gridSize.y);
    const sf::Vector2f viewSize{ maxSide * options.cellWidth, maxSide * options.cellWidth };
    gridView.setSize(viewSize);
    gridView.setCenter(viewSize.x / 2, viewSize.y / 2);
    zoomLevel = 1.0f;
}

sf::Vector2f GraphicsHandler::pixelToGridCoordinates(const sf::RenderWindow & window, const sf::Vector2i& pixel) const
{
    sf::View view{ gridView };
    view.zoom(zoomLevel);
    return window.mapPixelToCoords(pixel, view);
}
