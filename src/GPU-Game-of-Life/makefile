CPU := 0
target := life

objs := gpu/Kernel.o gpu/Update.o gui/Button.o gui/Game.o gui/GraphicsHandler.o gui/InputHandler.o Utils.o NoGui.o main.o

cudaroot  := /usr/bin
SFML_root := SFML-2.5.1-linux-gcc-x64

#----- C++ compilation options ------
gpp         := /usr/bin/g++ 
ccflags     := -Wall -Werror -std=c++14 -D UPDATE_ON_CPU=$(CPU)
cclib_paths := -L$(SFML_root)/lib
ccinc_paths := -I$(SFML_root)/include
cclibraries :=  $(SFML_root)/lib/libsfml-graphics.so.2.5.1 $(SFML_root)/lib/libsfml-window.so.2.5.1 $(SFML_root)/lib/libsfml-system.so.2.5.1

#----- CUDA compilation options -----
nvcc        := $(cudaroot)/nvcc
cuflags     :=
culib_paths := -L$(cudaroot)/$(CUDALIB) -L$(nv_sdk)/C/lib  
cuinc_paths := -I$(cudaroot)/include
culibraries := -lcuda -lcudart

lib_paths   := $(culib_paths) $(cclib_paths)
libraries   := $(culibraries) $(cclibraries)

all: life
    
life: $(objs)
	$(gpp) -o $(target) $(objs) $(ccflags) $(lib_paths) $(libraries)

gpu/Kernel.o: gpu/Kernel.cu
	$(nvcc) -o gpu/Kernel.o $(cuflags) -c gpu/Kernel.cu $(culib_paths) $(culibraries) $(cuinc_paths)
	
gpu/Update.o: gpu/Update.cu
	$(nvcc) -o gpu/Update.o $(cuflags) -c gpu/Update.cu $(culib_paths) $(culibraries) $(cuinc_paths)
	
gui/Button.o: gui/Button.cpp
	$(gpp) -o gui/Button.o $(ccflags) -c gui/Button.cpp $(ccinc_paths)
	
gui/Game.o: gui/Game.cpp
	$(gpp) -o gui/Game.o $(ccflags) -c gui/Game.cpp $(ccinc_paths)
	
gui/GraphicsHandler.o: gui/GraphicsHandler.cpp
	$(gpp) -o gui/GraphicsHandler.o $(ccflags) -c gui/GraphicsHandler.cpp $(ccinc_paths)
	
gui/InputHandler.o: gui/InputHandler.cpp
	$(gpp) -o gui/InputHandler.o $(ccflags) -c gui/InputHandler.cpp $(ccinc_paths)
	
Utils.o: Utils.cpp
	$(gpp) -o Utils.o $(ccflags) -c Utils.cpp $(ccinc_paths)
	
NoGui.o: NoGui.cpp
	$(gpp) -o NoGui.o $(ccflags) -c NoGui.cpp $(ccinc_paths)
	
main.o: main.cpp
	$(gpp) -o main.o $(ccflags) -c main.cpp $(ccinc_paths)

clean:
	rm -f $(target) $(objs)
