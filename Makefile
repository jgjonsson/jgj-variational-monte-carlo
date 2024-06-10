CXX:=g++

CXX_FLAGS_RELEASE:=-O3 
CXX_FLAGS_DEBUG:=-Wall -Wextra -g
CXX_FLAGS_EXTRALIBS:=-fopenmp -larmadillo
CXX_INCLUDES:=-I ~/anaconda3/include -I ~/anaconda3/include/eigen3/

HEADERS:=$(wildcard include/*.h)

APP_NAME = ${app}
SOURCES:=$(wildcard src/*.cpp)

OBJECTS:=$(SOURCES:.cpp=.o)
DEPENDENCIES:=$(OBJECTS:.o=.d)

all : release

release : $(OBJECTS)
	@mkdir -p $(dir bin/$(APP_NAME))
	$(CXX) $(CXX_INCLUDES) $(APP_NAME).cpp $^ -o bin/$(APP_NAME).out $(CXX_FLAGS_RELEASE) $(CXX_FLAGS_EXTRALIBS)

debug : $(OBJECTS)
	@mkdir -p $(dir bin/$(APP_NAME))
	$(CXX) $(CXX_INCLUDES) $(APP_NAME).cpp $^ -o bin/$(APP_NAME).out $(CXX_FLAGS_DEBUG) $(CXX_FLAGS_EXTRALIBS)

%.o : %.cpp 
	$(CXX) $(CXX_INCLUDES) -MMD -c $< -o $@

-include $(DEPENDENCIES)

clean:
	rm -f src/*.o src/*.d
	rm -f bin/$(APP_NAME).out
