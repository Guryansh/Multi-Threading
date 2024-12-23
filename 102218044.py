import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import psutil


def multiply_matrices(constant_matrix, matrix):
    return np.dot(constant_matrix, matrix)


# Execution time with different thread counts
def measure_execution_time(num_threads, matrices, constant_matrix):
    cpu_usages = []  # List to store CPU usage during execution
    with Pool(processes=num_threads) as pool:
        start_time = time.time()
        # Track CPU usage at the start
        cpu_usages.append(psutil.cpu_percent(interval=None))  # Initial CPU usage before computation
        pool.starmap(multiply_matrices, [(constant_matrix, m) for m in matrices])
        # Track CPU usage after computation
        cpu_usages.append(psutil.cpu_percent(interval=None))  # CPU usage after computation
        end_time = time.time()

    return (end_time - start_time) * 1000, cpu_usages  # Return both execution time and CPU usage


def main():
    matrix_size = 500
    num_matrices = 500
    constant_matrix = np.random.rand(matrix_size, matrix_size)
    matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(num_matrices)]

    # Number of threads to test
    num_threads_list = list(range(1, 9))

    times_taken = []
    cpu_usages_list = []

    # Matrix multiplication with different thread counts
    for num_threads in num_threads_list:
        print(f"\nTesting with {num_threads} threads...")
        # Measure execution time and CPU usage
        time_taken, cpu_usages = measure_execution_time(num_threads, matrices, constant_matrix)
        times_taken.append(time_taken)
        cpu_usages_list.append(cpu_usages)
        print(f"Time taken with {num_threads} threads: {time_taken:.2f} milliseconds")
        print(f"CPU usage before/after with {num_threads} threads: {cpu_usages}")

    # Execution time graph in milliseconds
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)  # Plot execution time
    plt.plot(num_threads_list, times_taken, marker='o', color='b', label="Execution Time")
    plt.xlabel("Number of Threads")
    plt.ylabel("Time Taken (Milliseconds)")
    plt.title("Execution Time vs. Number of Threads (Matrix Size: 500x500)")
    plt.xticks(num_threads_list)
    plt.legend()
    plt.grid(True)

    # CPU Usage graph
    cpu_usages_before = [usage[0] for usage in cpu_usages_list]  # Before computation
    cpu_usages_after = [usage[1] for usage in cpu_usages_list]  # After computation
    plt.subplot(2, 1, 2)  # Plot CPU Usage
    plt.plot(num_threads_list, cpu_usages_before, marker='o', color='r', label="CPU Usage Before")
    plt.plot(num_threads_list, cpu_usages_after, marker='s', color='g', label="CPU Usage After")
    plt.xlabel("Number of Threads")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Before and After Computation")
    plt.xticks(num_threads_list)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()


if __name__ == "__main__":
    main()
