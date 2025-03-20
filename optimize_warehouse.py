import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

def load_training_data():
    """Load the training data"""
    data = pd.read_csv('warehouse_data_files/training_data.csv')
    return data

def analyze_completion_time(data, warehouse_width, warehouse_height, total_orders):
    """
    Analyze completion time for a specific warehouse configuration and order count
    Returns optimal number of robots and workstations
    """
    # Filter data for the specified warehouse size
    warehouse_data = data[
        (data['warehouse_width'] == warehouse_width) & 
        (data['warehouse_height'] == warehouse_height)
    ].copy()
    
    if len(warehouse_data) == 0:
        print(f"No data available for warehouse size {warehouse_width}x{warehouse_height}")
        return None
    
    # Group by workstations and robots
    configs = warehouse_data.groupby(['num_workstations', 'num_robots'])
    
    results = []
    for (num_ws, num_robots), group in configs:
        # Calculate average completion time and normalize by order count
        avg_completion_time = group['completion_time'].mean() * (total_orders / group['total_orders'].mean())
        avg_orders_per_step = group['orders_per_step'].mean()
        
        results.append({
            'num_workstations': num_ws,
            'num_robots': num_robots,
            'completion_time': avg_completion_time,
            'orders_per_step': avg_orders_per_step
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal configuration
    optimal_config = results_df.loc[results_df['completion_time'].idxmin()]
    
    return results_df, optimal_config

def plot_completion_time_analysis(results_df, optimal_config, warehouse_width, warehouse_height, total_orders):
    """Create visualizations for completion time analysis"""
    
    # 1. Heatmap of completion time
    plt.figure(figsize=(12, 8))
    pivot_data = results_df.pivot(
        index='num_robots', 
        columns='num_workstations', 
        values='completion_time'
    )
    
    sns.heatmap(pivot_data, cmap='YlOrRd_r', annot=True, fmt='.0f')
    plt.title(f'Completion Time Heatmap\nWarehouse: {warehouse_width}x{warehouse_height}, Orders: {total_orders}')
    plt.xlabel('Number of Workstations')
    plt.ylabel('Number of Robots')
    plt.savefig('plots/completion_time_heatmap.png')
    plt.close()
    
    # 2. Completion Time vs Robots for different workstation counts
    plt.figure(figsize=(12, 6))
    for ws in results_df['num_workstations'].unique():
        ws_data = results_df[results_df['num_workstations'] == ws]
        plt.plot(ws_data['num_robots'], ws_data['completion_time'], 
                marker='o', label=f'{ws} workstations')
    
    plt.axvline(x=optimal_config['num_robots'], color='r', linestyle='--', 
                label=f'Optimal: {optimal_config["num_robots"]} robots')
    plt.axhline(y=optimal_config['completion_time'], color='r', linestyle='--',
                label=f'Min time: {optimal_config["completion_time"]:.0f} steps')
    
    plt.xlabel('Number of Robots')
    plt.ylabel('Completion Time (steps)')
    plt.title(f'Completion Time vs Number of Robots\nWarehouse: {warehouse_width}x{warehouse_height}, Orders: {total_orders}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/completion_time_vs_robots.png')
    plt.close()
    
    # 3. Orders per Step vs Robots
    plt.figure(figsize=(12, 6))
    for ws in results_df['num_workstations'].unique():
        ws_data = results_df[results_df['num_workstations'] == ws]
        plt.plot(ws_data['num_robots'], ws_data['orders_per_step'], 
                marker='o', label=f'{ws} workstations')
    
    plt.axvline(x=optimal_config['num_robots'], color='r', linestyle='--', 
                label=f'Optimal: {optimal_config["num_robots"]} robots')
    
    plt.xlabel('Number of Robots')
    plt.ylabel('Orders per Step')
    plt.title(f'Processing Rate vs Number of Robots\nWarehouse: {warehouse_width}x{warehouse_height}, Orders: {total_orders}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/orders_per_step_vs_robots.png')
    plt.close()

def analyze_multiple_sizes(data, warehouse_sizes, order_counts):
    """Analyze multiple warehouse sizes and order counts"""
    summary_results = []
    
    for width, height in warehouse_sizes:
        for orders in order_counts:
            print(f"\nAnalyzing warehouse {width}x{height} with {orders} orders...")
            results_df, optimal_config = analyze_completion_time(data, width, height, orders)
            
            if optimal_config is not None:
                summary_results.append({
                    'warehouse_width': width,
                    'warehouse_height': height,
                    'warehouse_area': width * height,
                    'total_orders': orders,
                    'optimal_workstations': optimal_config['num_workstations'],
                    'optimal_robots': optimal_config['num_robots'],
                    'min_completion_time': optimal_config['completion_time'],
                    'max_orders_per_step': optimal_config['orders_per_step']
                })
                
                # Generate plots for this configuration
                plot_completion_time_analysis(results_df, optimal_config, width, height, orders)
    
    return pd.DataFrame(summary_results)

def main():
    # Load data
    print("Loading training data...")
    data = load_training_data()
    
    # Define warehouse sizes and order counts to analyze
    warehouse_sizes = [(30, 50), (40, 70), (50, 85)]
    order_counts = [100, 200, 300]
    
    # Analyze all combinations
    print("Analyzing warehouse configurations...")
    summary = analyze_multiple_sizes(data, warehouse_sizes, order_counts)
    
    # Save summary results
    summary.to_csv('plots/optimization_summary.csv', index=False)
    
    # Print summary
    print("\nOptimal Configurations Summary:")
    print(summary.to_string(index=False))
    
    print("\nResults saved to:")
    print("- plots/optimization_summary.csv")
    print("- plots/completion_time_heatmap.png")
    print("- plots/completion_time_vs_robots.png")
    print("- plots/orders_per_step_vs_robots.png")

if __name__ == "__main__":
    main() 