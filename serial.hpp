#pragma once

#include <ctime>
#include <iomanip>

#include <fstream>
#include <dirent.h>

namespace neural
{
namespace serial
{

enum layer_type {
    linear_layer
};

struct layer_info
{
    layer_type type; // Type of layer
    int inputs; // Number of inputs
    int outputs; // Number of outputs

    layer_info() {}
    layer_info(layer_type t, int i, int o) : type(t), inputs(i), outputs(o) {}

    friend const bool operator!=(const layer_info& lhs, const layer_info& rhs)
    {
        return lhs.type    != rhs.type
            || lhs.inputs  != rhs.inputs
            || lhs.outputs != rhs.outputs;
    }
};

struct meta
{
    int fh = 0x4e4e5054; // TPNN (Templated Neural Network)
    int size; // File size
    int layers; // Number of layers

    friend const bool operator!=(const meta& lhs, const meta& rhs)
    {
        return lhs.fh     != rhs.fh
            || lhs.size   != rhs.size
            || lhs.layers != rhs.layers;
    }
};

template <typename Net>
std::string save(const Net &net,
                 const std::string &dir, const std::string &prefix)
{
    auto now = std::time(nullptr);
    std::stringstream filename_stream;
    filename_stream << std::put_time(
        std::localtime(&now), "%Y-%m-%d-%H-%M-%S.net");

    auto filename = dir + "/" + prefix + "-" + filename_stream.str();

    std::ofstream ofs(filename.c_str(), std::ios::binary);

    meta m;
    m.layers = Net::NumLayers;
    m.size = Net::NumLayers * sizeof(layer_info) + sizeof(meta) + Net::Size;

    ofs.write((char *)&m, sizeof(m));
    ofs.write((char *)&Net::info, sizeof(Net::info));

    ofs << net;

    return filename;
}

// Loads the specified file into the network
template <typename Net>
std::string load(Net &net, const std::string &filename)
{
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    
    if (!ifs) {
        std::cout << "Could not open the file '" << filename
                  << "'" << std::endl;
        return "";
    }

    // Read the meta
    meta m;
    ifs.read((char *)&m, sizeof(m));
    if (!ifs)
    {
        std::cout << "Could not read the file '" << filename
                  << "'" << std::endl;
        return "";
    }

    meta expected;
    expected.layers = Net::NumLayers;
    expected.size = Net::NumLayers * sizeof(layer_info) + sizeof(meta) + Net::Size;

    if (m.fh != expected.fh)
    {
        std::cout << "The file '" << filename
                  << "' is not of the correct format" << std::endl;
        return "";
    }

    if (m != expected)
    {
        std::cout << "The file '" << filename
                  << "' is not the correct size. ("
                  << m.size << " bytes, expected " << expected.size
                  << ")" << std::endl;
        return "";
    }

    decltype(Net::info) info;
    ifs.read((char *)&info, sizeof(info));
    if (!ifs)
    {
        std::cout << "Could not read the file '" << filename
                  << "'" << std::endl;
        return "";
    }

    if (utils::__ne(info, Net::info))
    {
        std::cout << "The '" << filename
                  << "' represents a network with different layers"
                  << " and cannot be read by this network" << std::endl;
        return "";
    }

    try
    {
        ifs >> net;
    }
    catch (int e)
    {
        std::cout << "There was an error loading the file '"
                  << filename << "'. (Error number: "
                  << e << ")" << std::endl;
        return "";
    }

    return filename;
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
        return load(net, dir + "/" + filename);

    return filename;
}

} // namespace serial
} // namespace neural
