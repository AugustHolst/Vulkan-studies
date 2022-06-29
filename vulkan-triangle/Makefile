CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi #lgflw requires the low-level linkers

VulkanTriangle: main.cpp
	g++ $(CFLAGS) -o VulkanTriangle main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTriangle
	./VulkanTriangle

clean:
	rm -f VulkanTriangle
