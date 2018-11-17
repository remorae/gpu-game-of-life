#pragma once

#include <functional>
#include <memory>
#include <string>

#include <SFML/Graphics.hpp>

class Button
{
private:
    sf::RectangleShape rectangle;
    std::unique_ptr<sf::Font> font = nullptr;
    std::unique_ptr<sf::Text> text = nullptr;
    std::function<void()> onClick;

public:
    Button(sf::RectangleShape rectangle, std::function<void()> onClick);

    void setText(sf::Text text, sf::Font font);
    void setPosition(float x, float y);

    void draw(sf::RenderWindow& window) const;
    void clicked() const { onClick(); }

    sf::FloatRect getGlobalBounds() const { return rectangle.getGlobalBounds(); }
};