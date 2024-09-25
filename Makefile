CXX = g++
CXXFLAGS = -O3 -Wall --std=c++20 -march=native

TARGET = L2sqr_simd_bench

all: $(TARGET)

$(TARGET): $(TARGET).cc
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TARGET).cc
clean:
	$(RM) $(TARGET)
