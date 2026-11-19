from tumor import run_simulation
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pathlib
import json
import time


def main():
    NUM_SIMULATIONS = 200
    MAX_WORKERS = 4
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output folder for the run
    folder_path = pathlib.Path(f"./output/{TIMESTAMP}")
    folder_path.mkdir(parents=True, exist_ok=True)

    results = []
    crashed = []

    # Start the simulation
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as process_executor:
        futures = {
            process_executor.submit(run_simulation, sim_num=i, timestamp=TIMESTAMP): i
            for i in range(NUM_SIMULATIONS)
        }
        for future in as_completed(futures):
            sim_num = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                # A worker raised an exception (not a numerical blow-up)
                crashed.append({"sim_num": sim_num + 1, "error": repr(e)})
                print(f"[Simulation {sim_num+1}] CRASHED: {e!r}")
    end_time = time.perf_counter()

    results.sort(key=lambda r: r["sim_num"])
    blown_up = [r for r in results if r["blew_up"]]
    survived = [r for r in results if not r["blew_up"]]

    print(f"\nFinished ({NUM_SIMULATIONS} simulations in {end_time - start_time:.2f} secs)")
    print(f"  Blew up:   {len(blown_up)}  -> sims {[r['sim_num'] for r in blown_up]}")
    print(f"  Survived:  {len(survived)}  -> sims {[r['sim_num'] for r in survived]}")
    if crashed:
        print(f"  Crashed:   {len(crashed)}  -> sims {[c['sim_num'] for c in crashed]}")

    if blown_up:
        print("\n=== Parameters of simulations that blew up ===")
        for r in blown_up:
            print(f"\nSimulation {r['sim_num']} (step {r['blowup_step']}, t = {r['blowup_time']:.6f})")
            for name, value in r["parameters"].items():
                print(f"    {name:12s} = {value:.6g}")

    # Save a run-level summary
    summary = {
        "timestamp": TIMESTAMP,
        "num_simulations": NUM_SIMULATIONS,
        "num_blown_up": len(blown_up),
        "blown_up": blown_up,
        "survived": [r["sim_num"] for r in survived],
        "crashed": crashed,
    }
    with open(folder_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary written to {folder_path / 'summary.json'}")


if __name__ == "__main__":
    main()
