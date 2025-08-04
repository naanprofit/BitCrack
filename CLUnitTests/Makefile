.RECIPEPREFIX := >

NAME=CLUnitTests
CPPSRC:=main.cpp

# Generated source from OpenCL kernels
CLTEST:=cltest.cpp

# Object files to link into the final test binary
OBJS:=$(CPPSRC:.cpp=.o) $(CLTEST:.cpp=.o)

all: $(NAME)

# Combine the OpenCL source files
cltest.cl: ../clMath/secp256k1.cl secp256k1test.cl
>cat $^ > $@

# Embed the combined OpenCL source into a C++ file
$(CLTEST): cltest.cl
>${BINDIR}/embedcl $< $@ _secp256k1_test_cl

# Compile C++ sources
%.o: %.cpp
>${CXX} -c $< ${INCLUDE} -I${OPENCL_INCLUDE} ${CXXFLAGS}

# Link objects into the unit test executable
$(NAME): $(OBJS)
>${CXX} -o clunittest.bin $(OBJS) ${LIBS} -L${OPENCL_LIB} -lclutil -lutil -lOpenCL
>mkdir -p $(BINDIR)
>cp clunittest.bin $(BINDIR)/clunittest

clean:
>rm -rf *.a *.o clunittest.bin cltest.cl cltest.cpp

.PHONY: all clean $(NAME)
