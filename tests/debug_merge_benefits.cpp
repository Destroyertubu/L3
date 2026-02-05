#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <algorithm>

// Simple linear regression
void fit_linear(const uint64_t* data, int start, int end, double& theta0, double& theta1) {
    int n = end - start;
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = start; i < end; i++) {
        double x = i - start;
        double y = static_cast<double>(data[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double dn = n;
    double det = dn * sum_xx - sum_x * sum_x;
    if (fabs(det) > 1e-10) {
        theta1 = (dn * sum_xy - sum_x * sum_y) / det;
        theta0 = (sum_y - theta1 * sum_x) / dn;
    } else {
        theta1 = 0;
        theta0 = sum_y / dn;
    }
}

int compute_max_error_bits(const uint64_t* data, int start, int end, double theta0, double theta1) {
    long long max_error = 0;
    for (int i = start; i < end; i++) {
        int local_idx = i - start;
        double pred = theta0 + theta1 * local_idx;
        long long pred_val = static_cast<long long>(round(pred));
        long long delta = static_cast<long long>(data[i]) - pred_val;
        max_error = std::max(max_error, std::abs(delta));
    }
    if (max_error == 0) return 0;
    int bits = 0;
    while ((1LL << bits) <= max_error) bits++;
    return bits + 1;
}

float compute_cost(int n, int bits) {
    float model_overhead = 64.0f;  // Same as GPU_MERGE_MODEL_OVERHEAD_BYTES
    float delta_bytes = static_cast<float>(n) * bits / 8.0f;
    return model_overhead + delta_bytes;
}

int main() {
    // Load first 10000 elements
    std::vector<uint64_t> data(10000);
    std::ifstream f("data/sosd/normal_200M_uint64.bin", std::ios::binary);
    if (!f) { std::cerr << "Cannot open file\n"; return 1; }
    f.read(reinterpret_cast<char*>(data.data()), 10000 * sizeof(uint64_t));
    f.close();
    
    std::cout << "Manual merge benefit analysis\n";
    std::cout << "============================\n\n";
    
    // Let's analyze the first 4 partitions (size 2048 each)
    int partition_size = 2048;
    
    for (int p = 0; p < 4; p++) {
        int start_a = p * partition_size;
        int end_a = (p + 1) * partition_size;
        int start_b = end_a;
        int end_b = std::min((p + 2) * partition_size, 10000);
        
        if (end_b <= start_b) break;
        
        // Fit partition A
        double theta0_a, theta1_a;
        fit_linear(data.data(), start_a, end_a, theta0_a, theta1_a);
        int bits_a = compute_max_error_bits(data.data(), start_a, end_a, theta0_a, theta1_a);
        float cost_a = compute_cost(end_a - start_a, bits_a);
        
        // Fit partition B  
        double theta0_b, theta1_b;
        fit_linear(data.data(), start_b, end_b, theta0_b, theta1_b);
        int bits_b = compute_max_error_bits(data.data(), start_b, end_b, theta0_b, theta1_b);
        float cost_b = compute_cost(end_b - start_b, bits_b);
        
        // Fit merged partition
        double theta0_c, theta1_c;
        fit_linear(data.data(), start_a, end_b, theta0_c, theta1_c);
        int bits_c = compute_max_error_bits(data.data(), start_a, end_b, theta0_c, theta1_c);
        float cost_c = compute_cost(end_b - start_a, bits_c);
        
        float separate_cost = cost_a + cost_b;
        float benefit = (separate_cost - cost_c) / separate_cost;
        
        std::cout << "Partition " << p << " (" << start_a << "-" << end_a << "):\n";
        std::cout << "  theta0_a=" << theta0_a << " theta1_a=" << theta1_a << " bits_a=" << bits_a << " cost_a=" << cost_a << "\n";
        std::cout << "Partition " << (p+1) << " (" << start_b << "-" << end_b << "):\n";
        std::cout << "  theta0_b=" << theta0_b << " theta1_b=" << theta1_b << " bits_b=" << bits_b << " cost_b=" << cost_b << "\n";
        std::cout << "Merged (" << start_a << "-" << end_b << "):\n";
        std::cout << "  theta0_c=" << theta0_c << " theta1_c=" << theta1_c << " bits_c=" << bits_c << " cost_c=" << cost_c << "\n";
        std::cout << "Benefit: " << (benefit * 100) << "% (threshold: 5%)\n";
        std::cout << "Should merge: " << (benefit >= 0.05 ? "YES" : "NO") << "\n\n";
    }
    
    // Now test the GPU version's O(1) statistics formula
    std::cout << "\n=== O(1) Statistics Formula Test ===\n";
    
    int start_a = 0, end_a = 2048;
    int start_b = 2048, end_b = 4096;
    
    // Manual computation of statistics for partition A
    double sum_x_a = 0, sum_y_a = 0, sum_xx_a = 0, sum_xy_a = 0;
    for (int i = start_a; i < end_a; i++) {
        double x = i - start_a;
        double y = static_cast<double>(data[i]);
        sum_x_a += x;
        sum_y_a += y;
        sum_xx_a += x * x;
        sum_xy_a += x * y;
    }
    
    // Manual computation of statistics for partition B
    double sum_x_b = 0, sum_y_b = 0, sum_xx_b = 0, sum_xy_b = 0;
    for (int i = start_b; i < end_b; i++) {
        double x = i - start_b;  // Local index
        double y = static_cast<double>(data[i]);
        sum_x_b += x;
        sum_y_b += y;
        sum_xx_b += x * x;
        sum_xy_b += x * y;
    }
    
    // Manual computation of statistics for merged partition (ground truth)
    double sum_x_gt = 0, sum_y_gt = 0, sum_xx_gt = 0, sum_xy_gt = 0;
    for (int i = start_a; i < end_b; i++) {
        double x = i - start_a;  // Merged local index
        double y = static_cast<double>(data[i]);
        sum_x_gt += x;
        sum_y_gt += y;
        sum_xx_gt += x * x;
        sum_xy_gt += x * y;
    }
    
    // O(1) formula: B's indices shift by n_a
    int n_a = end_a - start_a;
    int n_b = end_b - start_b;
    double dn_a = n_a;
    double dn_b = n_b;
    
    double sum_x_calc = sum_x_a + sum_x_b + dn_a * dn_b;
    double sum_y_calc = sum_y_a + sum_y_b;
    double sum_xx_calc = sum_xx_a + sum_xx_b + 2.0 * dn_a * sum_x_b + dn_a * dn_a * dn_b;
    double sum_xy_calc = sum_xy_a + sum_xy_b + dn_a * sum_y_b;
    
    std::cout << "Ground truth (full scan):\n";
    std::cout << "  sum_x=" << sum_x_gt << " sum_y=" << sum_y_gt << "\n";
    std::cout << "  sum_xx=" << sum_xx_gt << " sum_xy=" << sum_xy_gt << "\n";
    
    std::cout << "O(1) formula:\n";
    std::cout << "  sum_x=" << sum_x_calc << " sum_y=" << sum_y_calc << "\n";
    std::cout << "  sum_xx=" << sum_xx_calc << " sum_xy=" << sum_xy_calc << "\n";
    
    std::cout << "Difference:\n";
    std::cout << "  sum_x diff=" << fabs(sum_x_gt - sum_x_calc) << "\n";
    std::cout << "  sum_y diff=" << fabs(sum_y_gt - sum_y_calc) << "\n";
    std::cout << "  sum_xx diff=" << fabs(sum_xx_gt - sum_xx_calc) << "\n";
    std::cout << "  sum_xy diff=" << fabs(sum_xy_gt - sum_xy_calc) << "\n";
    
    return 0;
}
