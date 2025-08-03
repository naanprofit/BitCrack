NAME=PollardTests
CPPSRC=main.cpp
BINDIR?=.

all:
	${CXX} -o pollardtests.bin ${CPPSRC} ${CXXFLAGS}
	mkdir -p $(BINDIR)
	cp pollardtests.bin $(BINDIR)/pollardtests

clean:
	rm -f pollardtests.bin
