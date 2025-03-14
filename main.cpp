#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>   // for std::atoi
#include <chrono>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef double real_t;

enum class MulMode {
    SERIAL,
    SERIAL_TRANSPOSED,
    OMP,
    OMP_TRANSPOSED,
    OMP_SIMD
};

void transpose(const std::vector<real_t>& B,
               std::vector<real_t>&       Btrans, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Btrans[j*N + i] = B[i*N + j];
        }
    }
}

void matMul(const std::vector<real_t>& A,
            const std::vector<real_t>& B,
            const std::vector<real_t>& Btrans,
            std::vector<real_t>&       C,
            int N,
            MulMode mode)
{
    switch (mode) {
        case MulMode::SERIAL:
        {
            // ---- Serial (no OpenMP) ----
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    real_t sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        }
        case MulMode::SERIAL_TRANSPOSED:
        {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    real_t sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i*N + k] * Btrans[j*N + k];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        }
        case MulMode::OMP:
        {
            // ---- OpenMP parallel on outer loops ----
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    real_t sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        }
        case MulMode::OMP_TRANSPOSED:
        {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    real_t sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i*N + k] * Btrans[j*N + k];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        }
        case MulMode::OMP_SIMD:
        {
            // ---- OpenMP parallel + SIMD on the inner loop ----
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    real_t sum = 0.0;
                    // Vectorize the k-loop
                    #pragma omp simd reduction(+:sum)
                    for (int k = 0; k < N; ++k) {
                        sum += A[i*N + k] * Btrans[j*N + k];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
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

#ifdef _OPENMP
    int numThreads = omp_get_max_threads();
#else
    int numThreads = 1;
#endif

    double totalTimeSerial    = 0.0;
    double totalTimeSerialTrans = 0.0;
    double totalTimeOmp       = 0.0;
    double totalTimeOmpTransposed = 0.0;
    double totalTimeOmpSimd   = 0.0;

    // Allocate matrices
    std::vector<real_t> A(N*N);
    std::vector<real_t> B(N*N);
    std::vector<real_t> Btrans(N*N);
    // Result matrices
    std::vector<real_t> C_serial(N*N, 0);
    std::vector<real_t> C_serial_transposed(N*N, 0);
    std::vector<real_t> C_omp(N*N, 0);
    std::vector<real_t> C_ompTransposed(N*N, 0);
    std::vector<real_t> C_ompSimd(N*N, 0);

    // -------- Warm-up iterations (not timed) --------
    std::cout << "Performing 10 warm-up iterations (all modes)...\n";
    for (int w = 0; w < 10; ++w) {
        // Randomize A and B each time
        for (int i = 0; i < N*N; ++i) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        transpose(B, Btrans, N);

        // Serial
        matMul(A, B, Btrans, C_serial, N, MulMode::SERIAL);
        // Serial Transposed
        matMul(A, B, Btrans, C_serial_transposed, N, MulMode::SERIAL_TRANSPOSED);
        // OMP
        matMul(A, B, Btrans, C_omp, N, MulMode::OMP);
        // OMP_TRANSPOSED
        matMul(A, B, Btrans, C_ompTransposed, N, MulMode::OMP_TRANSPOSED);
        // OMP_SIMD
        matMul(A, B, Btrans, C_ompSimd, N, MulMode::OMP_SIMD);

        // correctness check
        double maxDiff = 0.0;
        for (int i = 0; i < N*N; ++i) {
            double diff1 = std::fabs(C_serial[i] - C_omp[i]);
            double diff2 = std::fabs(C_serial[i] - C_ompSimd[i]);
            double diff3 = std::fabs(C_serial[i] - C_ompTransposed[i]);
            double diff4 = std::fabs(C_serial[i] - C_serial_transposed[i]);
            maxDiff = std::max({maxDiff, diff1, diff2, diff3, diff4});
        }
        if (maxDiff > 1e-10) {
            std::cerr << "Warm-up warning: max diff = " << maxDiff
                      << " in iteration " << (w+1) << "\n";
        }
    }
    std::cout << "Warm-up complete.\n\n";

    // -------- Timed iterations --------
    for (int r = 0; r < reps; ++r) {
        // Randomize A and B
        for (int i = 0; i < N*N; ++i) {
            A[i] = dist(gen);
            B[i] = dist(gen);
        }

        transpose(B, Btrans, N);

        // --- Serial ---
        {
            auto t1 = std::chrono::steady_clock::now();
            matMul(A, B, Btrans, C_serial, N, MulMode::SERIAL);
            auto t2 = std::chrono::steady_clock::now();
            double elapsedMs = std::chrono::duration<double,std::milli>(t2 - t1).count();
            totalTimeSerial += elapsedMs;
        }

        // --- Serial Transposed ---
        {
            auto t1 = std::chrono::steady_clock::now();
            matMul(A, B, Btrans, C_serial_transposed, N, MulMode::SERIAL_TRANSPOSED);
            auto t2 = std::chrono::steady_clock::now();
            double elapsedMs = std::chrono::duration<double,std::milli>(t2 - t1).count();
            totalTimeSerialTrans += elapsedMs;
        }

        // --- OMP ---
        {
            auto t1 = std::chrono::steady_clock::now();
            matMul(A, B, Btrans, C_omp, N, MulMode::OMP);
            auto t2 = std::chrono::steady_clock::now();
            double elapsedMs = std::chrono::duration<double,std::milli>(t2 - t1).count();
            totalTimeOmp += elapsedMs;
        }

        // --- OMP Transposed ---
        {
            auto t1 = std::chrono::steady_clock::now();
            matMul(A, B, Btrans, C_ompTransposed, N, MulMode::OMP_TRANSPOSED);
            auto t2 = std::chrono::steady_clock::now();
            double elapsedMs = std::chrono::duration<double,std::milli>(t2 - t1).count();
            totalTimeOmpTransposed += elapsedMs;
        }

        // --- OMP + SIMD ---
        {
            auto t1 = std::chrono::steady_clock::now();
            matMul(A, B, Btrans, C_ompSimd, N, MulMode::OMP_SIMD);
            auto t2 = std::chrono::steady_clock::now();
            double elapsedMs = std::chrono::duration<double,std::milli>(t2 - t1).count();
            totalTimeOmpSimd += elapsedMs;
        }

        // Compare all three results
        double maxDiff = 0.0;
        for (int i = 0; i < N*N; ++i) {
            double diff1 = std::fabs(C_serial[i] - C_omp[i]);
            double diff2 = std::fabs(C_serial[i] - C_ompSimd[i]);
            double diff3 = std::fabs(C_serial[i] - C_ompTransposed[i]);
            double diff4 = std::fabs(C_serial[i] - C_serial_transposed[i]);
            maxDiff = std::max({maxDiff, diff1, diff2, diff3, diff4});
        }
        if (maxDiff > 1e-10) {
            std::cerr << "Warning: Results differ (max diff = "
                      << maxDiff << ") on repetition " << (r+1) << "!\n";
        }
    }

    // Print results
    std::cout << "========== Summary ==========\n";
    std::cout << "Matrix size: " << N << "\n";
    std::cout << "Repetitions (timed): " << reps << "\n";
    std::cout << "OpenMP threads: " << numThreads << "\n";

    std::cout << "\nAverage time (ms)  per " << reps << " iteration:\n";
    std::cout << "  Serial            : " << totalTimeSerial / reps  << " ms\n";
    std::cout << "  Serial Transposed : " << totalTimeSerialTrans / reps << " ms\n";
    std::cout << "  OMP               : " << totalTimeOmp / reps    << " ms\n";
    std::cout << "  OMP Transposed    : " << totalTimeOmpTransposed / reps << " ms\n";
    std::cout << "  OMP + SIMD        : " << totalTimeOmpSimd / reps << " ms\n";
    std::cout << "=============================\n";

    return 0;
}
