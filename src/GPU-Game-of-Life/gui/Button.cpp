#include "Button.h"

Button::Button(sf::RectangleShape rectangle, std::function<void()> onClick) :
    rectangle{ std::move(rectangle) },
    onClick{ std::move(onClick) }
{
}

void Button::setText(sf::Text text, sf::Font font)
{
    this->text = std::make_unique<sf::Text>(std::move(text));
    this->font = std::make_unique<sf::Font>(std::move(font));
    this->text->setFont(*this->font);
}

void Button::setPosition(float x, float y)
{
    rectangle.setPosition(x, y);
    if (text)
    {
        text->setPosition(x, y);
    }
}

void Button::draw(sf::RenderWindow& window) const
{
    window.draw(rectangle);
    if (text && font)
    {
        window.draw(*text);
    }
}
