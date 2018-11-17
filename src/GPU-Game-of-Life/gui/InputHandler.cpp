#include "InputHandler.h"

#include <SFML/Graphics.hpp>

#include <iostream>

#include "Button.h"
#include "Game.h"
#include "GraphicsHandler.h"

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

InputHandler::InputHandler(InputConfig options) :
    options(std::move(options))
{
}

void InputHandler::handleEvents(sf::RenderWindow& window, Game& game, GraphicsHandler& graphicsHandler)
{
    previousMousePosition = currentMousePosition;
    mouseJustPressed.clear();
    mouseJustReleased.clear();

    sf::Event event;
    bool mouseMoved = false;
    while (window.pollEvent(event))
    {
        switch (event.type)
        {
            default:
                break;

            case sf::Event::Closed:
                window.close();
                break;

            case sf::Event::KeyPressed:
                //handleKeyDown(event);
                break;

            case sf::Event::MouseWheelScrolled:
                graphicsHandler.handleZoom(event.mouseWheelScroll.delta);
                break;

            case sf::Event::MouseButtonPressed:
                onMousePress(event, game, graphicsHandler);
                break;

            case sf::Event::MouseButtonReleased:
                onMouseRelease(event);
                break;

            case sf::Event::MouseMoved:
                if (options.outputMousePosition)
                {
                    std::cout << "X: " << event.mouseMove.x << ", Y: " << event.mouseMove.y << "\n";
                }

                currentMousePosition.x = static_cast<float>(event.mouseMove.x);
                currentMousePosition.y = static_cast<float>(event.mouseMove.y);
                mouseMoved = true;
                break;
        }
    }

    if (mouseMoved)
    {
        onMouseMove(event, game, graphicsHandler);
    }
}

bool InputHandler::mouseButtonDown(const sf::Mouse::Button& button) const
{
    return std::find(currentMouseDown.begin(), currentMouseDown.end(), button) != currentMouseDown.end();
}

bool InputHandler::mouseButtonPressed(const sf::Mouse::Button& button) const
{
    return std::find(mouseJustPressed.begin(), mouseJustPressed.end(), button) == mouseJustPressed.end();
}

bool InputHandler::mouseButtonReleased(const sf::Mouse::Button& button) const
{
    return std::find(mouseJustReleased.begin(), mouseJustReleased.end(), button) == mouseJustReleased.end();
}

namespace
{
    void setEditMode(const sf::Vector2<size_t>& position, CellEditMode& mode, const Game& game)
    {
        mode = (game.getCell(position.x, position.x) == 1) ? CellEditMode::OFF : CellEditMode::ON;
    }

    void editCell(const sf::Vector2<size_t>& position, const CellEditMode& mode, Game& game)
    {
        game.setCell(position.x, position.y, (mode == CellEditMode::ON) ? 1 : 0);
    }

    bool mouseOverCell(const Game& game, const GraphicsHandler& graphicsHandler)
    {
        const sf::Vector2f position = game.pixelToGridCoordinates() / graphicsHandler.getCellWidth();
        return position.x >= 0
            && position.y >= 0
            && position.x < game.getWidth()
            && position.y < game.getHeight();
    }

    sf::Vector2<size_t> getCellAtMousePosition(const Game& game, const GraphicsHandler& graphicsHandler)
    {
        const sf::Vector2f position = game.pixelToGridCoordinates() / graphicsHandler.getCellWidth();
        return { static_cast<size_t>(position.x), static_cast<size_t>(position.y) };
    }
}

void InputHandler::onMouseMove(const sf::Event& event, Game& game, GraphicsHandler& graphicsHandler)
{
    if (isPanning)
    {
        graphicsHandler.handlePan({ previousMousePosition.x - currentMousePosition.x, previousMousePosition.y - currentMousePosition.y });
    }

    if (editMode != CellEditMode::NONE
        && mouseButtonDown(sf::Mouse::Button::Left)
        && mouseOverCell(game, graphicsHandler))
    {
        if (editMode != CellEditMode::NONE)
        {
            editCell(getCellAtMousePosition(game, graphicsHandler), editMode, game);
        }
    }
}

void InputHandler::onMousePress(const sf::Event& event, Game& game, const GraphicsHandler& graphicsHandler)
{
    currentMouseDown.push_back(event.mouseButton.button);
    mouseJustPressed.push_back(event.mouseButton.button);

    if (event.mouseButton.button == sf::Mouse::Middle)
    {
        setPanning(true);
    }

    if (event.mouseButton.button == sf::Mouse::Left)
    {
        const bool buttonClicked = checkButtons(event, game.getButtons());
        if (!buttonClicked && mouseOverCell(game, graphicsHandler))
        {
            setEditMode(getCellAtMousePosition(game, graphicsHandler), editMode, game);
            editCell(getCellAtMousePosition(game, graphicsHandler), editMode, game);
        }
    }
}

void InputHandler::onMouseRelease(const sf::Event& event)
{
    std::remove(currentMouseDown.begin(), currentMouseDown.end(), event.mouseButton.button);
    mouseJustReleased.push_back(event.mouseButton.button);

    if (event.mouseButton.button == sf::Mouse::Middle)
    {
        setPanning(false);
    }
    if (event.mouseButton.button == sf::Mouse::Left)
    {
        editMode = CellEditMode::NONE;
    }
}
