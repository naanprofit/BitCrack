NAME=PollardTests
CPPSRC=main.cpp

all:
	${CXX} -o pollardtests.bin ${CPPSRC} ../KeyFinder/PollardEngine.cpp ${INCLUDE} ${CXXFLAGS} ${LIBS} -lsecp256k1 -laddressutil -lcryptoutil -lutil
	mkdir -p $(BINDIR)
	cp pollardtests.bin $(BINDIR)/pollardtests

clean:
	rm -f pollardtests.bin
