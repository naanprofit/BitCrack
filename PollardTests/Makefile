NAME=PollardTests
CPPSRC=main.cpp
BINDIR?=.

all:
	${CXX} -o pollardtests.bin ${CPPSRC} ${INCLUDE} ${CXXFLAGS} ${LIBS} -laddressutil -lsecp256k1 -lcryptoutil
	mkdir -p $(BINDIR)
	cp pollardtests.bin $(BINDIR)/pollardtests

clean:
	rm -f pollardtests.bin
