SHELL = /bin/sh
CC = /usr/bin/mpicc.mpich #my pc
CC1 = mpicc #for other users
CXXFLAGS = -Wall -Wextra

SRC= ex5_sequential.c \
		 ex5_overlapping.c

EXE = $(SRC:.c=.x)

# eliminate default suffixes
.SUFFIXES:
SUFFIXES =

# just consider our own suffixes
.SUFFIXES: .c .x

all: $(EXE)

.PHONY: all

%.x: %.c
	$(CC) $< -o $@ $(CXXFLAGS)

#$(EXE): $(SRC)
#	$(CC) $(SRC) -o $(EXE)

#$(EXE2): $(SRC2)
#	$(CXX) $(SRC2) -o $(EXE2)

clean:
	@rm -rf $(EXE)

.PHONY: clean all
