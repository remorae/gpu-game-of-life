#include "Button.h"

Button::Button(sf::RectangleShape rectangle, std::function<void()> onClick) :
    rectangle{ std::move(rectangle) },
    onClick{ std::move(onClick) }
{
}

namespace
{
    void centerTextInRectangle(const sf::RectangleShape& rectangle, sf::Text* text)
    {
        if (text)
        {
            const sf::FloatRect buttonBounds = rectangle.getGlobalBounds();
            const sf::FloatRect textBounds = text->getLocalBounds();
            text->setPosition(buttonBounds.left + buttonBounds.width / 2 - textBounds.width / 2,
                              buttonBounds.top);
        }
    }

    void resizeRectFromText(sf::RectangleShape& rectangle, const sf::Text& text, const sf::Vector2f& padding)
    {
        const sf::FloatRect& textBounds = text.getLocalBounds();
        rectangle.setSize({ padding.x * 2 + textBounds.width, padding.y * 2 + textBounds.height });
    }
}

void Button::setText(sf::Text text, sf::Font font)
{
    this->text = std::make_unique<sf::Text>(std::move(text));
    this->font = std::make_unique<sf::Font>(std::move(font));
    this->text->setFont(*this->font);
    if (resizeToText)
    {
        resizeRectFromText(this->rectangle, *(this->text), padding);
    }
    centerTextInRectangle(this->rectangle, this->text.get());
}

void Button::setPosition(float x, float y)
{
    rectangle.setPosition(x, y);
    centerTextInRectangle(this->rectangle, this->text.get());
}

void Button::draw(sf::RenderWindow& window) const
{
    window.draw(rectangle);
    if (text && font)
    {
        window.draw(*text);
    }
}
