CXX = g++
CFLAGS = -g
# CFLAGS = -pg -O3 -MD

SRCS = RMCRTnoInterpolationFSKR.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc BinarySearchTree.cc\


OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTnoInterpolationFSKRL5 : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o RMCRTnoInterpolationFSKRL5

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d  RMCRTnoInterpolationFSKRL5 *.out

-include $(SRCS:.cc=.d)
