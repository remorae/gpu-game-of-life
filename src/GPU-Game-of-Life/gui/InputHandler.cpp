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
    previousMouseDown = currentMouseDown;
    currentMouseDown.clear();

    sf::Event event;
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
                currentMouseDown.push_back(event.mouseButton.button);

                if (event.mouseButton.button == sf::Mouse::Middle)
                    setPanning(true);

                if (event.mouseButton.button == sf::Mouse::Left)
                {

                    const bool buttonClicked = checkButtons(event, game.getButtons());
                    if (!buttonClicked)
                    {
                        const auto x = event.mouseButton.x; //?
                        const auto y = event.mouseButton.y; //?

                        if (std::find(previousMouseDown.begin(), previousMouseDown.end(), sf::Mouse::Left) == previousMouseDown.end())
                        {
                            //toggleMode = (game.getCell(x, y) == 1) ? ToggleMode::OFF : ToggleMode::ON;
                        }

                        //game.setCell(x, y, (toggleMode == ToggleMode::ON) ? 1 : 0);
                    }
                }
                break;

            case sf::Event::MouseButtonReleased:
                if (event.mouseButton.button == sf::Mouse::Middle)
                    setPanning(false);

                if (event.mouseButton.button == sf::Mouse::Left)
                    toggleMode = ToggleMode::NONE;

                break;

            case sf::Event::MouseMoved:
                if ()
                std::cout << "X: " << event.mouseMove.x << ", Y: " << event.mouseMove.y << "\n";
                currentMousePosition.x = static_cast<float>(-event.mouseMove.x);
                currentMousePosition.y = static_cast<float>(-event.mouseMove.y);
                if (isPanning)
                    graphicsHandler.handlePan({ currentMousePosition.x - previousMousePosition.x, currentMousePosition.y - previousMousePosition.y });
                break;
        }
    }
}