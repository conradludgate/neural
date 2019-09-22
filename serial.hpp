#include <ctime>
#include <iomanip>

#include <fstream>

#include <dirent.h>

namespace neural
{
namespace serial
{

template <typename Net>
std::string save(const Net &net, const std::string &dir, const std::string &prefix)
{
    auto now = std::time(nullptr);
    std::stringstream filename_stream;
    filename_stream << std::put_time(std::localtime(&now), "%Y-%m-%d-%H-%M-%S.net");

    auto filename = dir + "/" + prefix + "-" + filename_stream.str();

    std::ofstream ofs(filename.c_str(), std::ios::binary);

    ofs << net;

    return filename;
}

// Loads the specified file into the network
template <typename Net>
void load(Net &net, const std::string &filename)
{
    std::ifstream ifs(filename.c_str(), std::ios::binary);

    ifs >> net;
}

// Update to use filesystem when I can test it
template <typename Net>
std::string load(Net &net, const std::string &dir, const std::string &prefix)
{
    auto pre_length = prefix.length();

    std::string filename = "";

    // Open dir
    auto dirp = opendir(dir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL)
    {
        // Get file name
        std::string name(dp->d_name);

        // If file doesn't match the prefix, get the next file
        if (prefix.compare(name.substr(0, pre_length)) != 0)
            continue;

        // See if this file is older, if so, set that instead
        if (filename.compare(name) < 0)
            filename = name;
    }

    // Close dir
    (void)closedir(dirp);

    // Load file if one was found
    if (filename.length() != 0)
        load(net, filename);

    // Return file name
    return filename;
}

} // namespace serial
} // namespace neural
