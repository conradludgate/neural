#
# Compiler flags
#
CXX = g++
CXXFLAGS = -std=c++17 -I ../../.. -I /usr/include/eigen3

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