NAME=CudaKeySearchDevice
CPPSRC:=$(wildcard *.cpp)
CUSRC:=$(wildcard *.cu)

.RECIPEPREFIX := ;

all:    cuda

cuda:
;for file in ${CPPSRC} ; do\
;;${NVCC} -c $$file -o $$file".o" ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE};\
;done

;for file in ${CUSRC} ; do\
;;${NVCC} -c $$file -o $$file".o" ${NVCCFLAGS} ${INCLUDE} -I${CUDA_INCLUDE} -I${CUDA_MATH};\
;done

;${NVCC} -dlink -o cuda_libs.o *.o -lcudadevrt -lcudart

;ar rvs ${LIBDIR}/lib$(NAME).a *.o

clean:
;rm -f *.o
;rm -f *.a

