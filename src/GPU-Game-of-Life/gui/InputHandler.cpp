#include "InputHandler.h"

#include <SFML/Graphics.hpp>

#include <iostream>

#include "Button.h"
#include "Game.h"
#include "GraphicsHandler.h"

InputHandler::InputHandler(InputConfig options) :
    options(std::move(options))
{
}

void InputHandler::handleEvents(sf::RenderWindow& window, Game& game, GraphicsHandler& graphics)
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
                switch (event.key.code)
                {
                    case sf::Keyboard::Left:
                        game.pan(sf::Vector2i{ -1, 0 });
                        break;
                    case sf::Keyboard::Right:
                        game.pan(sf::Vector2i{ 1, 0 });
                        break;
                    case sf::Keyboard::Up:
                        game.pan(sf::Vector2i{ 0, -1 });
                        break;
                    case sf::Keyboard::Down:
                        game.pan(sf::Vector2i{ 0, 1 });
                        break;
                    default:
                        break;
                }
                break;

            case sf::Event::MouseWheelScrolled:
                graphics.handleZoom(event.mouseWheelScroll.delta);
                break;

            case sf::Event::MouseButtonPressed:
                onMousePress(event, game);
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
        const sf::Vector2f distance{ previousMousePosition.x - currentMousePosition.x, previousMousePosition.y - currentMousePosition.y };
        game.handleMouseMove(isPanning, distance);
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

sf::Vector2i InputHandler::getWindowMousePosition(const sf::RenderWindow& window) const
{
    return sf::Mouse::getPosition(window);
}

void InputHandler::onMousePress(const sf::Event& event, Game& game)
{
    currentMouseDown.push_back(event.mouseButton.button);
    mouseJustPressed.push_back(event.mouseButton.button);

    if (event.mouseButton.button == sf::Mouse::Middle)
    {
        setPanning(true);
    }

    if (event.mouseButton.button == sf::Mouse::Left)
    {
        game.handlePrimaryClick(event);
    }
}

void InputHandler::onMouseRelease(const sf::Event& event)
{
    currentMouseDown.erase(std::remove(currentMouseDown.begin(), currentMouseDown.end(), event.mouseButton.button));
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
