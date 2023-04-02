
#include "../include/file_io.h"

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>


void one_columns_to_csv(const std::string& fpath, const std::vector<double>& x, const std::string& separator, bool index, size_t digits)
{
    std::ofstream file(fpath);
    for (size_t i = 0; i < x.size(); ++i)
    {
        file << std::setprecision(digits);
        if (index)
            file << i << separator;
        file << x[i] << '\n';
    }
}

void two_columns_to_csv(const std::string& fpath, const std::vector<double>& x, const std::vector<double>& y, const std::string& separator, bool index, size_t digits)
{
    std::ofstream file(fpath);
    if (x.size() != y.size())
    {
        throw std::invalid_argument("x and y must have the same size");
    }
    for (size_t i = 0; i < x.size(); ++i)
    {
        file << std::setprecision(digits);
        if (index)
            file << i << separator;
        file << x[i] << separator << y[i] << '\n';
    }
}

void three_columns_to_csv(const std::string& fpath, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, const std::string& separator, bool index, size_t digits)
{
    std::ofstream file(fpath);
    if (x.size() != y.size() || x.size() != z.size())
    {
        throw std::invalid_argument("x,y and z must have the same size");
    }
    for (size_t i = 0; i < x.size(); ++i)
    {
        file << std::setprecision(digits);
        if (index)
            file << i << separator;
        file << x[i] << separator << y[i] << separator << z[i]<< '\n';
    }
}
