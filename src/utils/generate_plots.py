import pandas as pd
import matplotlib.pyplot as plt
import os
from config.paths import PLOT_DIR

def generate_plots(csv_path):
    print(f"Loading stats from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter out warmup if needed (or just plot everything and mark it)
    # df = df[df['is_warmup'] == False] 

    # Define report plots directory - USER REQUEST: Save alongside stats/metadata
    output_dir = os.path.dirname(csv_path)
    # output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))), "report", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Runtime Breakdown (Stacked)
    # Columns: rv_build_time, rtv_build_time, assignment_time, rebalancing_time
    runtime_cols = ['rv_build_time', 'rtv_build_time', 'assignment_time', 'rebalancing_time']
    
    plt.figure(figsize=(10, 6))
    plt.stackplot(df['sim_time'], df[runtime_cols].T, labels=['RV Build', 'RTV Build', 'ILP Solve', 'Rebalancing'])
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Runtime (s)')
    plt.title('Computational Runtime Breakdown')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'runtime_breakdown.png'))
    plt.close()
    print("Generated runtime_breakdown.png")

    # 2. Average Delay Over Time (Previously Wait Time)
    plt.figure(figsize=(10, 6))
    if 'average_delay' in df.columns:
        plt.plot(df['sim_time'], df['average_delay'], marker='o', label='Avg Delay')
        plt.title('Average Delay per Batch (Cost)')
        plt.ylabel('Delay (Cost)')
    else:
        # Fallback for old CSVs
        plt.plot(df['sim_time'], df['average_wait_time'], marker='o', label='Avg Wait Time')
        plt.title('Average Wait Time per Batch')
        plt.ylabel('Wait Time (s)')
        
    plt.xlabel('Simulation Time (s)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'wait_time_over_time.png')) # Keeping same filename for compatibility or change? 
    # Let's change filename to be correct
    plt.savefig(os.path.join(output_dir, 'average_delay_over_time.png'))
    plt.close()
    print("Generated average_delay_over_time.png")

    # 3. Fleet Utilization
    plt.figure(figsize=(10, 6))
    plt.plot(df['sim_time'], df['active_vehicles_count'], label='Active Vehicles', color='green')
    plt.plot(df['sim_time'], df['idle_vehicles_count'], label='Idle Vehicles', color='red')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Vehicle Count')
    plt.title('Fleet Utilization Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fleet_utilization.png'))
    plt.close()
    print("Generated fleet_utilization.png")

    # 4. Occupancy / Trip Sizes (Aggregation)
    # Sum up columns like trips_size_1, trips_size_2, etc.
    size_cols = [c for c in df.columns if 'trips_size_' in c]
    if size_cols:
        total_counts = df[size_cols].sum()
        # Clean labels: trips_size_1 -> 1 Pax
        labels = [c.replace('trips_size_', '') + ' Pax' for c in total_counts.index]
        
        plt.figure(figsize=(6, 6))
        plt.pie(total_counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Trip Sizes (Occupancy)')
        plt.savefig(os.path.join(output_dir, 'occupancy_distribution.png'))
        plt.close()
        print("Generated occupancy_distribution.png")

    # 5. Service Rate (Cumulative)
    # newly_served_count vs requests_count (requests count is snapshot size, need cumulative inputs?)
    # For now, let's just plot serving rate per batch
    plt.figure(figsize=(10, 6))
    plt.plot(df['sim_time'], df['newly_served_count'], marker='x', color='purple')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Requests Served per Batch')
    plt.title('Service Throughput')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'service_throughput.png'))
    plt.close()
    print("Generated service_throughput.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = os.path.join(PLOT_DIR, "simulation_stats.csv")
        
    if os.path.exists(csv_file):
        generate_plots(csv_file)
    else:
        print(f"File not found: {csv_file}")
