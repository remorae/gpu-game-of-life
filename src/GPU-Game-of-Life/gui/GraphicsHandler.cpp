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

GraphicsHandler::GraphicsHandler(GraphicsConfig options, size_t gridWidth, size_t gridHeight) :
    options(std::move(options))
{
    resetCamera(gridWidth, gridHeight);
    gridView.setViewport(sf::FloatRect{ 0.0f, 0.0f, 1.0f, options.gridViewportHeight });
}

void GraphicsHandler::drawGrid(sf::RenderWindow& window, const Game& game) const
{
    const size_t width = game.getWidth();
    const size_t height = game.getHeight();
        
    const float lineWidth = 0.0f;
    const float totalLineWidth = (width + 1) * lineWidth;
    const float totalLineHeight = (height + 1) * lineWidth;
    const float totalWidth = totalLineWidth + options.gridElementWidth * width;
    const float totalHeight = totalLineHeight + options.gridElementWidth * height;

    sf::View view{ gridView };
    view.zoom(zoomLevel);
    window.setView(view);
    window.clear(options.backgroundColor);
        
    if (options.drawLines)
    {
        // Horizontal lines
        for (int row = 0; row <= height; ++row)
        {
            const float startY = row * (options.gridElementWidth + lineWidth);
            DrawUtils::drawLine(window, sf::Vector2f{ 0.0f, startY }, sf::Vector2f{ totalWidth, startY });
        }

        // Vertical lines
        for (int col = 0; col <= width; ++col)
        {
            const float startX = col * (options.gridElementWidth + lineWidth);
            DrawUtils::drawLine(window, sf::Vector2f{ startX, 0.0f }, sf::Vector2f{ startX, totalHeight });
        }
    }

    // Cells
    sf::RectangleShape cell{ sf::Vector2f{ options.gridElementWidth, options.gridElementWidth } };
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            cell.setFillColor((game.getCell(col, row)) ? options.aliveCellColor : options.deadCellColor);
            cell.setPosition(sf::Vector2f{ col * (options.gridElementWidth + lineWidth), row * (options.gridElementWidth + lineWidth) });

            window.draw(cell);
        }
    }
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

void GraphicsHandler::resetCamera(size_t gridWidth, size_t gridHeight)
{
    const float width = gridWidth * options.gridElementWidth;
    const float height = gridHeight * options.gridElementWidth;
    gridView.setSize(width, height);
    gridView.setCenter(width / 2, height / 2);
    zoomLevel = 1.0f;
}

sf::Vector2f GraphicsHandler::pixelToGridCoordinates(const sf::RenderWindow& window) const
{
    const sf::Vector2i pixelPos = sf::Mouse::getPosition(window);
    sf::View view{ gridView };
    view.zoom(zoomLevel);
    return window.mapPixelToCoords(pixelPos, view);
}
