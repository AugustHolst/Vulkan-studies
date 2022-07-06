CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi #lgflw requires the low-level linkers


SOURCES = main.cpp shaderHelper.cpp

VulkanTriangle: main.cpp
	g++ $(CFLAGS) -o VulkanTriangle $(SOURCES) $(LDFLAGS)

.PHONY: test clean

test: VulkanTriangle
	./VulkanTriangle

clean:
	rm -f VulkanTriangle
