import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt


def multiply_matrices(constant_matrix, matrix):
    return np.dot(constant_matrix, matrix)


# Execution time with different thread counts
def measure_execution_time(num_threads, matrices, constant_matrix):
    with Pool(processes=num_threads) as pool:
        start_time = time.time()
        pool.starmap(multiply_matrices, [(constant_matrix, m) for m in matrices])
        end_time = time.time()
    return (end_time - start_time) * 1000  # Convert to milliseconds


def main():
    matrix_size = 500
    num_matrices = 500
    constant_matrix = np.random.rand(matrix_size, matrix_size)
    matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(num_matrices)]

    # Number of threads to test
    num_threads_list = list(range(1, 9))

    times_taken = []

    # Matrix multiplication with different thread counts
    for num_threads in num_threads_list:
        print(f"\nTesting with {num_threads} threads...")
        # Measure execution time
        time_taken = measure_execution_time(num_threads, matrices, constant_matrix)
        times_taken.append(time_taken)
        print(f"Time taken with {num_threads} threads: {time_taken:.2f} milliseconds")

    # Execution time graph in milliseconds
    plt.figure(figsize=(8, 6))
    plt.plot(num_threads_list, times_taken, marker='o', color='b', label="Execution Time")
    plt.xlabel("Number of Threads")
    plt.ylabel("Time Taken (Milliseconds)")
    plt.title("Execution Time vs. Number of Threads (Matrix Size: 500x500)")
    plt.xticks(num_threads_list)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
