from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def process_data(label, delay):
    time.sleep(delay)
    return f"Task {label} finished in {delay} secs"

def run_simulation(sim_num):
    return f"Simulation {sim_num} finished"

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=10) as executor:
        future_results = {
            executor.submit(
                run_simulation,
                sim_num
            ): sim_num for sim_num in range(100)
        }

        for simulation in as_completed(future_results):
            run = future_results[simulation]
            try:
               print(run)
            except Exception as e:
                print(f"Task {run} generated an exception: {e}")


    # tasks = [
    #     ('A', 2), 
    #     ('B', 0.5),
    #     ('C', 1)
    # ]
    
    # with ProcessPoolExecutor(max_workers=5) as executor:
    #     # Schedule all tasks; this creates a dictionary mapping Futures to labels
    #     future_to_label = {
    #         executor.submit(
    #             process_data, 
    #             label, delay
    #         ): label for label, delay in tasks
    #     }
        
    #     # As each process finishes, yield its completed Future
    #     for future in as_completed(future_to_label):
    #         label = future_to_label[future]
    #         try:
    #             data = future.result()  # Get result (or raises exception if task failed)
    #             print(data)
    #         except Exception as e:
    #             print(f"Task {label} generated an exception: {e}")
