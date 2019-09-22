#
# Compiler flags
#
CXX = g++
CXXFLAGS = -std=c++17 -I include

ifeq ($(OS),Windows_NT)
    # Window's not currently supported
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
		CXXFLAGS += -I /usr/include/eigen3
    endif
    ifeq ($(UNAME_S),Darwin)
        CXXFLAGS += -I /usr/local/include/eigen3
    endif
endif

#
# Project files
#
SRCDIR = src
SRCS = $(wildcard $(SRCDIR)/*)
OBJS = $(SRCS:$(SRCDIR)/%.cpp=%.o)
EXE  = mnist

#
# Release build settings
#
RELDIR = build
RELOBJS = $(addprefix $(RELDIR)/, $(OBJS))
RELCXXFLAGS = -O3 -DNDEBUG

.PHONY: all clean debug prep release remake

# Default build
all: prep release

#
# Release rules
#
release: $(EXE)

$(EXE): $(RELOBJS)
	$(CXX) $(CXXFLAGS) $(RELCXXFLAGS) -o $(EXE) $^

$(RELDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(RELCXXFLAGS) -o $@ $<

#
# Other rules
#
prep:
	@mkdir -p $(RELDIR)

remake: clean all

clean:
	rm -f $(RELOBJS)