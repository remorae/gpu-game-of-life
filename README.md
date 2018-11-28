# GPU Game of Life
A graphical depiction of Conway’s Game of Life written in C++.

## Features
* [x] Pause and un-pause the simulation
* [x] Click on cells to toggle their state
* [x] Pan the grid
* [x] Zoom in and out
* [x] Clear the existing grid
* [x] Randomize the existing grid
* [x] Display current UPS/FPS
* [x] Pre-built test cases
* [ ] Stretch Goal: Control the speed of the simulation
* [ ] Stretch Goal: Deploy “templates” of commonly-known entities in the Game of Life, e.g. gliders

## Implementation Details
All update operations will occur on the GPU. This will allow for huge grid sizes to be updated quickly. Ideally, along with tile-based kernel configuration, we will divide the grid into “chunks” and only update portions that have any alive cells for maximum performance.

## Demonstration

![Demonstration](https://cloud.alexplagman.com/index.php/s/SGQkPGqy943neQp/download)

## Compilation

Compilation:

The makefile is in src/GPU-Game-of-Life/

`make CPU=1` to compile for the CPU; `make CPU=0` or just `make` to compile for the GPU

The project depends on [SFML 2.5.1](https://www.sfml-dev.org/download/sfml/2.5.1/).
Edit `SFML_root` in the makefile to point to wherever you place the compiled files.

You may also need to update `cclibraries`, `gpp`, and `cudaroot` if compiling on a system other than Linux.

## Running the Program:

`./life blockWidth blockHeight gridWidth gridHeight [numPasses] [p] [t<1|2>]`

Parameters: 
- `blockWidth`, `blockHeight`, `gridWidth`, and `gridHeight` must be positive integers.
- Non-square grids will be stretched to fit the window if using the GUI.
- `blockWidth` and `blockHeight` will be ignored if running on the CPU.
- If `numPasses` (a positive integer) is given, the GUI will be disabled. Instead, a starting grid will be generated and the program will execute exactly `numPasses` updates (or kernel calls). Note: Timings are averaged by ignoring the first iteration since it is abnormally slow on the GPU.
- If `p` is given and `numPasses` is given, the program will print the grid after each iteration.
- If `t#` is given, the program will no longer start with a random grid. Instead, it will contain a pre-defined test setup. This can be used to verify CPU / GPU correctness. The grid may be expanded automatically beyond the given arguments if it is too small.
  - t1: The grid will contain a [light-weight space ship](http://www.conwaylife.com/wiki/Lightweight_spaceship) that should move horizontally across the grid.  
  - t2: The grid will contain a [Gosper glider gun](http://www.conwaylife.com/wiki/Gosper_glider_gun).
