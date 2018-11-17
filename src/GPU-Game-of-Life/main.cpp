#include <SFML/Graphics.hpp>

#include "gui/Game.h"
#include "gui/GraphicsHandler.h"

int main()
{
    Game game{ GameConfig{}, GraphicsConfig{} };
    game.run();

    return 0;
}