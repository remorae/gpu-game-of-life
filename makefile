# vim:comments=
#
# I don't like the Cuda SDK makefiles. Here's my own! This makefile should
# allow you to build Cuda programs independent of the NVidia Cuda SDK. Its
# simplicity makes it easy for you to put objects, binaries, etc where you want
# them. There is even some boiler-plate Makefile magic that you can copy
# into other Makefiles.
#
# Mark Murphy 2008
#
# http://www.eecs.berkeley.edu/~mjmurphy/makefile
#
# PJR Modified extensively for fev.
# PJR Modified again for 582

HOSTTYPE = $(shell uname -m)
BITS := $(if $(filter x86_64 x86_64-linux , $(HOSTTYPE)),64,32)

# Fill in the name of the output binary here
target    := life 


#build cuda lib directory name
ifeq ($(BITS),64)
        CUDALIB := lib64
else
        CUDALIB := lib
endif




# List of sources with .c, .cu, and .cc extensions
# PLEASE INCLUDE the target source file (e.g. Server.cpp)
sources   := \
				main.cu\
				GameOfLife.cu\
				Methods.cu\


 
# Other things that need to be built, e.g. .cubin files
extradeps := 

# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

opt  ?= -m$(BITS)
dbg  ?= -g -O0
cuda_dbg ?= -G $(dbg)
warn ?= -Wall -Werror 

# This is where the cuda runtime libraries and includes can be found

cudaroot  := /usr/local/cuda-8.0

# Tony: location of NVIDIA SDK
nv_sdk := $(cudaroot)


#----- C compilation options ------
gcc        := /usr/bin/gcc
cflags     += $(opt) $(dbg) $(warn)
clib_paths :=

cinc_paths := -I $(cudaroot)/samples/common/inc
clibraries := 


#----- C++ compilation options ------
gpp         := /usr/bin/g++ 
ccflags     += $(opt) $(dbg) $(warn)
cclib_paths := 
ccinc_paths := -I$(cudaroot)/samples/common/inc -I$(nv_sdk)/samples/common/inc 
cclibraries :=  


#----- CUDA compilation options -----
nvcc        := $(cudaroot)/bin/nvcc
cuflags     += $(opt) $(cuda_dbg) 
culib_paths := -L$(cudaroot)/$(CUDALIB) -L$(nv_sdk)/C/lib -L$(nv_sdk)/samples/common/lib/$(OSTYPE)  
cuinc_paths := -I$(cudaroot)/include -I$(nv_sdk)/samples/common/inc 
culibraries := -lcuda -lcudart 


lib_paths   := $(culib_paths) $(cclib_paths) $(clib_paths)
libraries   := $(culibraries) $(cclibraries) $(clibraries)


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

csources  := $(filter %.c ,$(sources))
ccsources := $(filter %.cc,$(sources))
cppsources := $(filter %.cpp,$(sources))
cusources := $(filter %.cu,$(sources))

# This code generates a list of object files by replacing filename extensions

objects := $(patsubst %.c,%.o ,$(csources))  \
           $(patsubst %.cu,%.o,$(cusources)) \
           $(patsubst %.cc,%.o,$(ccsources)) \
           $(patsubst %.cpp,%.o,$(cppsources))


#----- Build rules ------

# foo:
#	echo $(CUDALIB)

all: $(target) 



$(target): $(objects) 
	$(gpp)  -o $@ $(objects) $(ccflags) $(lib_paths) $(libraries)




%.o: %.cu
	$(nvcc) -arch=sm_30 -c $< $(cuflags) $(cuinc_paths) -o $@

%.cubin: %.cu
	$(nvcc) -cubin $(cuflags) $(cuinc_paths) $^

%.o: %.cc
	$(gpp) -c $< $(ccflags) $(ccinc_paths) -o $@

%.o: %.cpp
	$(gpp) -c $< $(ccflags) $(ccinc_paths) -o $@

%.o: %.c
	$(gcc) -c $< $(cflags) $(cinc_paths) -o $@

clean:
	rm -f $(target) $(objects) *.mk  *.combine *.vcg *.sibling 




#----- Dependency Generation -----
#
# If a particular set of sources is non-empty, then have rules for
# generating the necessary dep files.
#

ccdep := ccdep.mk
cdep  := cdep.mk
cudep := cudep.mk
cppdep := cppdep.mk

depfiles =

ifneq ($(ccsources),)

depfiles += $(ccdep)
$(ccdep): $(ccsources)
	$(gpp) -MM $(ccsources) > $(ccdep)

else

$(ccdep):

endif

ifneq ($(cusources),)

depfiles += $(cudep)
$(cudep):
	$(gpp) -MM -x c++ $(cuinc_paths) $(ccinc_paths) $(cusources) > $(cudep)

else

$(cudep):

endif

ifneq ($(csources),)

depfiles += $(cdep)
$(cdep): $(csources)
	$(gcc) -MM -x c $(csources) > $(cdep)

else

$(cdep):

endif


ifneq ($(cppsources),)

depfiles += $(cppdep)
$(cppdep): $(cppsources)
	$(gpp) -MM -x c++ $(ccinc_paths) $(cppsources)  > $(cppdep)

else

$(cppdep):

endif




.PHONY: dep
dep: $(depfiles)


ifneq ($(MAKECMDGOALS),dep)
 ifneq ($(MAKECMDGOALS),clean)
 
  ifneq ($(ccsources),)
   include $(ccdep)
  endif
  
  ifneq ($(cusources),)
   include $(cudep)
  endif
  
  ifneq ($(csources),)
   include $(cdep)
  endif
  
  ifneq ($(cppsources),)
   include $(cppdep)
  endif

 endif
endif
