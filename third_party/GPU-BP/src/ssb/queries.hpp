#pragma once

#include <string>

namespace ssb {

struct RunOptions {
    std::string data_dir;
    int         runs = 1;
};

void run_q11(const RunOptions& opt);
void run_q12(const RunOptions& opt);
void run_q13(const RunOptions& opt);

void run_q21(const RunOptions& opt);
void run_q22(const RunOptions& opt);
void run_q23(const RunOptions& opt);

void run_q31(const RunOptions& opt);
void run_q32(const RunOptions& opt);
void run_q33(const RunOptions& opt);
void run_q34(const RunOptions& opt);

void run_q41(const RunOptions& opt);
void run_q42(const RunOptions& opt);
void run_q43(const RunOptions& opt);

} // namespace ssb

