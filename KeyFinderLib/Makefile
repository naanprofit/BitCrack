NAME=keyfinder
CPPSRC:=$(wildcard *.cpp)

all:
ifeq ($(CPU),1)
	for file in ${CPPSRC} ; do \
	${CXX} -c $$file ${INCLUDE} ${CXXFLAGS}; \
	done
else
	for file in ${CPPSRC} ; do \
	${CXX} -c $$file ${INCLUDE} -I${CUDA_INCLUDE} ${CXXFLAGS}; \
	done
endif

	ar rvs ${LIBDIR}/lib$(NAME).a *.o

clean:
	rm -f *.o *.cu.o
	rm -f *.a

