import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

def load_training_data():
    """Load the training data"""
    data = pd.read_csv('warehouse_data_files/training_data.csv')
    return data

def analyze_capacity(data):
    """
    Analyze the relationship between number of robots and throughput
    for different warehouse configurations
    """
    # Group data by warehouse size and workstations
    configs = data.groupby(['warehouse_width', 'warehouse_height', 'num_workstations'])
    
    results = []
    for (width, height, num_ws), group in configs:
        # For each configuration, find the relationship between robots and throughput
        robot_throughput = group.groupby('num_robots')['orders_per_step'].mean()
        
        # Find the optimal number of robots (maximum throughput)
        optimal_robots = robot_throughput.idxmax()
        max_throughput = robot_throughput.max()
        
        results.append({
            'warehouse_width': width,
            'warehouse_height': height,
            'warehouse_area': width * height,
            'num_workstations': num_ws,
            'optimal_robots': optimal_robots,
            'max_throughput': max_throughput,
            'robots_per_area': optimal_robots / (width * height),
            'throughput_per_area': max_throughput / (width * height)
        })
    
    return pd.DataFrame(results)

def plot_capacity_analysis(results):
    """Create visualizations for capacity analysis"""
    
    # 1. Optimal Robots vs Warehouse Area for different workstation counts
    plt.figure(figsize=(12, 6))
    for ws in results['num_workstations'].unique():
        ws_data = results[results['num_workstations'] == ws]
        plt.scatter(ws_data['warehouse_area'], ws_data['optimal_robots'], 
                   label=f'{ws} workstations', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(ws_data['warehouse_area'], ws_data['optimal_robots'], 1)
        p = np.poly1d(z)
        plt.plot(ws_data['warehouse_area'], p(ws_data['warehouse_area']), '--', alpha=0.5)
    
    plt.xlabel('Warehouse Area (cells)')
    plt.ylabel('Optimal Number of Robots')
    plt.title('Optimal Robot Count vs Warehouse Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/optimal_robots.png')
    plt.close()
    
    # 2. Maximum Throughput vs Warehouse Area
    plt.figure(figsize=(12, 6))
    for ws in results['num_workstations'].unique():
        ws_data = results[results['num_workstations'] == ws]
        plt.scatter(ws_data['warehouse_area'], ws_data['max_throughput'], 
                   label=f'{ws} workstations', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(ws_data['warehouse_area'], ws_data['max_throughput'], 1)
        p = np.poly1d(z)
        plt.plot(ws_data['warehouse_area'], p(ws_data['warehouse_area']), '--', alpha=0.5)
    
    plt.xlabel('Warehouse Area (cells)')
    plt.ylabel('Maximum Throughput (orders/step)')
    plt.title('Maximum Throughput vs Warehouse Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/max_throughput.png')
    plt.close()
    
    # 3. Robots per Area vs Throughput per Area
    plt.figure(figsize=(12, 6))
    plt.scatter(results['robots_per_area'], results['throughput_per_area'], 
                c=results['num_workstations'], cmap='viridis')
    plt.colorbar(label='Number of Workstations')
    plt.xlabel('Robots per Area')
    plt.ylabel('Throughput per Area')
    plt.title('Efficiency Metrics: Normalized by Warehouse Area')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/efficiency_metrics.png')
    plt.close()
    
    # 4. Create a summary table
    summary = results.groupby('num_workstations').agg({
        'optimal_robots': ['mean', 'std'],
        'max_throughput': ['mean', 'std'],
        'robots_per_area': ['mean', 'std'],
        'throughput_per_area': ['mean', 'std']
    }).round(4)
    
    summary.to_csv('plots/capacity_summary.csv')
    
    return summary

def main():
    # Load data
    print("Loading training data...")
    data = load_training_data()
    
    # Analyze capacity
    print("Analyzing warehouse capacity...")
    results = analyze_capacity(data)
    
    # Generate visualizations and summary
    print("Generating visualizations...")
    summary = plot_capacity_analysis(results)
    
    # Print summary statistics
    print("\nSummary Statistics by Number of Workstations:")
    print(summary)
    
    print("\nResults saved to:")
    print("- plots/optimal_robots.png")
    print("- plots/max_throughput.png")
    print("- plots/efficiency_metrics.png")
    print("- plots/capacity_summary.csv")

if __name__ == "__main__":
    main() 