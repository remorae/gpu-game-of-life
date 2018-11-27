#pragma once

#include <functional>
#include <memory>
#include <string>

#include <SFML/Graphics.hpp>

class Button
{
private:
    sf::RectangleShape rectangle;
    std::unique_ptr<sf::Font> font;
    std::unique_ptr<sf::Text> text;
    std::function<void()> onClick;
    bool resizeToText{ true };
    sf::Vector2f padding{ 5.0f, 5.0f };

public:
    Button(sf::RectangleShape rectangle, std::function<void()> onClick);

    const sf::Font* getFont() const { return this->font.get(); };
    const sf::Text* getText() const { return this->text.get(); };
    void setText(sf::Text text, sf::Font font);
    void setPosition(float x, float y);

    void draw(sf::RenderWindow& window) const;
    void clicked() const { onClick(); }

    sf::FloatRect getGlobalBounds() const { return rectangle.getGlobalBounds(); }
};