#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ssb/queries.hpp"
#include "ssb/ssb_dataset.hpp"

int main(int argc, char** argv) {
    try {
        int         query = 0;
        std::string data_dir = ssb::default_data_dir();
        int         runs = 1;

        auto usage = [&]() {
            std::cerr << "Usage: ssb_planner [--query Q] [--data_dir DIR] [--runs N]\n"
                      << "  --query/-q: 11,12,13,21,22,23,31,32,33,34,41,42,43 (0 = all)\n"
                      << "  --data_dir: SSB binary column directory (default: " << data_dir << ")\n"
                      << "  --runs/-r:  number of runs per query (default: 1)\n";
        };

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                usage();
                return 0;
            }
            if (arg == "-q" || arg == "--query") {
                if (i + 1 >= argc) {
                    usage();
                    throw std::runtime_error("missing value for --query");
                }
                query = std::atoi(argv[++i]);
                continue;
            }
            if (arg == "--data_dir") {
                if (i + 1 >= argc) {
                    usage();
                    throw std::runtime_error("missing value for --data_dir");
                }
                data_dir = argv[++i];
                continue;
            }
            if (arg == "-r" || arg == "--runs") {
                if (i + 1 >= argc) {
                    usage();
                    throw std::runtime_error("missing value for --runs");
                }
                runs = std::atoi(argv[++i]);
                continue;
            }
            usage();
            throw std::runtime_error("unknown argument: " + arg);
        }

        if (runs < 1 || runs > 100) {
            throw std::runtime_error("--runs out of range (1..100)");
        }
        if (query < 0) {
            throw std::runtime_error("--query must be >= 0");
        }

        ssb::RunOptions opt;
        opt.data_dir = data_dir;
        opt.runs = runs;

        auto run_one = [&](int q) {
            switch (q) {
                case 11: ssb::run_q11(opt); break;
                case 12: ssb::run_q12(opt); break;
                case 13: ssb::run_q13(opt); break;
                case 21: ssb::run_q21(opt); break;
                case 22: ssb::run_q22(opt); break;
                case 23: ssb::run_q23(opt); break;
                case 31: ssb::run_q31(opt); break;
                case 32: ssb::run_q32(opt); break;
                case 33: ssb::run_q33(opt); break;
                case 34: ssb::run_q34(opt); break;
                case 41: ssb::run_q41(opt); break;
                case 42: ssb::run_q42(opt); break;
                case 43: ssb::run_q43(opt); break;
                default:
                    throw std::runtime_error("unsupported query id: " + std::to_string(q));
            }
        };

        if (query == 0) {
            const int queries[] = {11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43};
            for (int q : queries) {
                run_one(q);
            }
        } else {
            run_one(query);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

