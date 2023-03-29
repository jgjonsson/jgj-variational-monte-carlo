CXX:=g++

CXX_FLAGS_RELEASE:=-O3 
CXX_FLAGS_DEBUG:=-Wall -Wextra -g

HEADERS:=$(wildcard include/*.h)

APP_NAME = ${app}
SOURCES:=$(wildcard src/*.cpp)

OBJECTS:=$(SOURCES:.cpp=.o)
DEPENDENCIES:=$(OBJECTS:.o=.d)

all : release

release : $(OBJECTS)
	@mkdir -p $(dir bin/$(APP_NAME))
	$(CXX) $(APP_NAME).cpp $^ -o bin/$(APP_NAME).out $(CXX_FLAGS_RELEASE)

debug : $(OBJECTS)
	@mkdir -p $(dir bin/$(APP_NAME))
	$(CXX) $(APP_NAME).cpp $^ -o bin/$(APP_NAME).out $(CXX_FLAGS_DEBUG)

%.o : %.cpp 
	$(CXX) -MMD -c $< -o $@

-include $(DEPENDENCIES)

clean:
	rm -f src/*.o src/*.d
	rm -f bin/$(APP_NAME).out