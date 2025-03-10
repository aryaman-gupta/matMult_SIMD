#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>   // for std::atoi
#include <chrono>    // for timing
#include <cmath>     // for std::fabs
#ifdef _OPENMP
#include <omp.h>
#endif

typedef double real_t;

// Matrix multiplication (NxN), stored in row-major 1D arrays:
// element (i, j) at index i*N + j.
// 'useParallel' toggles OpenMP usage for the outer loop(s).
void matMul(const std::vector<real_t>& A,
            const std::vector<real_t>& B,
            std::vector<real_t>&       C,
            int N,
            bool useParallel)
{
    if (useParallel) {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                real_t sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                real_t sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_size> <num_repetitions>\n";
        return 1;
    }

    int N    = std::atoi(argv[1]);
    int reps = std::atoi(argv[2]);

    if (N <= 0 || reps <= 0) {
        std::cerr << "Error: matrix_size and num_repetitions must be positive.\n";
        return 1;
    }

    // Prepare random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_t> dist(0.0, 1.0);

    std::vector<real_t> A(N*N);
    std::vector<real_t> B(N*N);
    std::vector<real_t> C_parallel(N*N, 0);
    std::vector<real_t> C_serial(N*N, 0);

    // Track total times (in milliseconds)
    double totalParallelTime = 0.0;
    double totalSerialTime   = 0.0;

    for (int r = 0; r < reps; ++r) {
        // Fill A and B with random values
        for (int i = 0; i < N*N; ++i) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        // --- Parallel multiply ---
        auto startPar = std::chrono::steady_clock::now();
        matMul(A, B, C_parallel, N, true);
        auto endPar   = std::chrono::steady_clock::now();
        double elapsedParMs =
            std::chrono::duration<double, std::milli>(endPar - startPar).count();
        totalParallelTime += elapsedParMs;

        // --- Serial multiply ---
        auto startSer = std::chrono::steady_clock::now();
        matMul(A, B, C_serial, N, false);
        auto endSer   = std::chrono::steady_clock::now();
        double elapsedSerMs =
            std::chrono::duration<double, std::milli>(endSer - startSer).count();
        totalSerialTime += elapsedSerMs;

        // Compare results for correctness
        double maxDiff = 0.0;
        for (int i = 0; i < N*N; ++i) {
            double diff = std::fabs(C_parallel[i] - C_serial[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        if (maxDiff > 1e-10) {
            std::cerr << "Warning: Results differ (max diff = " << maxDiff
                      << ") on repetition " << (r+1) << "!\n";
        }
    }

#ifdef _OPENMP
    int numThreads = omp_get_max_threads();
#else
    int numThreads = 1;
#endif

    std::cout << "\n========== Summary ==========\n";
    std::cout << "Matrix size: "      << N    << "\n";
    std::cout << "Repetitions: "      << reps << "\n";
    std::cout << "OpenMP threads: "   << numThreads << "\n";
    std::cout << "Total parallel time (ms):     " << totalParallelTime << "\n";
    std::cout << "Total non-parallel time (ms): " << totalSerialTime   << "\n";
    std::cout << "============================\n";

    return 0;
}
