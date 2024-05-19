
#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <vector>

std::vector<double> csv_to_one_column(const std::string& fpath);

/// @brief Write a csv file with the given data column by column, 1 column.
/// @param fpath The path to the file to write
/// @param x First column of data
/// @param separator The separator to use between columns. Defaults to ",".
/// @param index If true, write an index column. Default is false.
/// @param digits The number of digits to write for each number. Default is 7.
void one_columns_to_csv(const std::string &fpath, const std::vector<double> &x, const std::string &separator = ",", bool index = false, size_t digits = 7);
/// @brief Write a csv file with the given data column by column, 2 columns.
/// @param fpath The path to the file to write
/// @param x First column of data
/// @param y Second column of data
/// @param separator The separator to use between columns. Defaults to ",".
/// @param index If true, write an index column. Default is false.
/// @param digits The number of digits to write for each number. Default is 7.
void two_columns_to_csv(const std::string &fpath, const std::vector<double> &x, const std::vector<double> &y, const std::string &separator = ",", bool index = false, size_t digits = 7);
/// @brief Write a csv file with the given data column by column, 3 columns.
/// @param fpath The path to the file to write
/// @param x First column of data
/// @param y Second column of data
/// @param z Third column of data
/// @param separator The separator to use between columns. Defaults to ",".
/// @param index If true, write an index column. Default is false.
/// @param digits The number of digits to write for each number. Default is 7.
void three_columns_to_csv(const std::string& fpath, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, const std::string &separator = ",", bool index = false, size_t digits = 7);

#endif