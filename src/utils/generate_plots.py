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
    # Columns: time_rv, time_rtv, time_assign, time_rebal
    runtime_cols = ['time_rv', 'time_rtv', 'time_assign', 'time_rebal']
    
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
    # Check for new column name
    if 'avg_total_delay' in df.columns:
        # Filter: Only plot batches where requests were actually served
        df_served = df[df['requests_delivered'] > 0]
        if not df_served.empty:
            plt.plot(df_served['sim_time'], df_served['avg_total_delay'], marker='o', label='Avg Delay')
        else:
             plt.plot([], [], label='No Data') # Handle case with no served requests

        plt.title('Average Delay per Batch (Cost)')
        plt.ylabel('Delay (Cost)')
    else:
        # Fallback (unlikely needed if schema is strict)
        pass
        
    plt.xlabel('Simulation Time (s)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'average_delay_over_time.png'))
    plt.close()
    print("Generated average_delay_over_time.png")

    # 3. Fleet Utilization
    plt.figure(figsize=(10, 6))
    plt.plot(df['sim_time'], df['vehicles_active'], label='Active Vehicles', color='green')
    plt.plot(df['sim_time'], df['vehicles_idle'], label='Idle Vehicles', color='red')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Vehicle Count')
    plt.title('Fleet Utilization Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fleet_utilization.png'))
    plt.close()
    print("Generated fleet_utilization.png")

    # 4. Occupancy / Trip Sizes (Aggregation)
    # Use ASSIGNED trips for occupancy distribution
    size_cols = [c for c in df.columns if 'assigned_trips_size_' in c]
    if size_cols:
        total_counts = df[size_cols].sum()
        # Clean labels: assigned_trips_size_1 -> 1 Pax
        labels = [c.replace('assigned_trips_size_', '') + ' Pax' for c in total_counts.index]
        
        # Filter out 0 counts
        filtered_counts = []
        filtered_labels = []
        for count, label in zip(total_counts, labels):
            if count > 0:
                filtered_counts.append(count)
                filtered_labels.append(label)

        if filtered_counts:
            plt.figure(figsize=(6, 6))
            plt.pie(filtered_counts, labels=filtered_labels, autopct='%1.1f%%', startangle=140)
            plt.title('Distribution of Trip Sizes (Occupancy)')
            plt.savefig(os.path.join(output_dir, 'occupancy_distribution.png'))
            plt.close()
            print("Generated occupancy_distribution.png")

    # 5. Service Rate (Cumulative)
    plt.figure(figsize=(10, 6))
    plt.plot(df['sim_time'], df['requests_delivered'], marker='x', color='purple')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Requests Delivered per Batch')
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
