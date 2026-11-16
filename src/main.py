from tumor import *
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import pathlib
import time

def main():
    NUM_SIMULATIONS = 1
    MAX_WORKERS = 5
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output folder for the run
    folder_path = pathlib.Path(f"./output/{TIMESTAMP}")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Start the simulation
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
        for sim_num in range(NUM_SIMULATIONS):
            process_executor.submit(
                run_simulation, 
                sim_num=sim_num, 
                timestamp=TIMESTAMP
            )
    end_time = time.perf_counter()
    print(f"Finished ({NUM_SIMULATIONS} simulations completed in {end_time - start_time} secs)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)