#include <vector>
#include <fstream>

class ShaderHelper
{
public:
    static std::vector<char> readFile(const std::string& filename);
};
