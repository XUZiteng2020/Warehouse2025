import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def load_and_prepare_data():
    """Load and prepare the training data"""
    data = pd.read_csv('warehouse_data_files/training_data.csv')
    
    # Calculate orders per area ratio
    data['orders_per_area'] = data['total_orders'] / (data['warehouse_width'] * data['warehouse_height'])
    
    # Calculate robots per area ratio
    data['robots_per_area'] = data['num_robots'] / (data['warehouse_width'] * data['warehouse_height'])
    
    return data

def analyze_completion_time_vs_robots(data, target_orders=1000, robot_range=(30, 150)):
    """Analyze relationship between completion time and number of robots for specific order count"""
    
    # Filter data for target order count (allow some variation) and 2 workstations
    order_margin = 0.1  # Allow 10% variation in order count
    filtered_data = data[
        (data['total_orders'] >= target_orders * (1 - order_margin)) &
        (data['total_orders'] <= target_orders * (1 + order_margin)) &
        (data['num_robots'] >= robot_range[0]) &
        (data['num_robots'] <= robot_range[1]) &
        (data['num_workstations'] == 2)  # Only 2 workstations
    ].copy()
    
    if len(filtered_data) == 0:
        print(f"No data available for {target_orders} orders")
        return None
    
    # Calculate mean and std for each robot count
    grouped_data = filtered_data.groupby('num_robots').agg({
        'completion_time': ['mean', 'std'],
        'orders_per_step': ['mean', 'std']
    }).reset_index()
    
    # Calculate marginal benefit (reduction in completion time per additional robot)
    grouped_data['marginal_benefit'] = -(grouped_data['completion_time']['mean'].diff() / 5)  # per 5 robots
    grouped_data['marginal_benefit_per_robot'] = grouped_data['marginal_benefit'] / 5  # per single robot
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 2])
    
    # Plot 1: Completion Time vs Number of Robots
    ax1.errorbar(grouped_data['num_robots'], 
                grouped_data['completion_time']['mean'],
                yerr=grouped_data['completion_time']['std'],
                fmt='bo-', linewidth=2, markersize=8, alpha=0.8,
                label='2 workstations', capsize=5)
    
    # Add trend line
    z = np.polyfit(grouped_data['num_robots'], grouped_data['completion_time']['mean'], 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(grouped_data['num_robots'].min(), grouped_data['num_robots'].max(), 100)
    ax1.plot(x_smooth, p(x_smooth), 'r--', alpha=0.5, label='Trend')
    
    # Find optimal configuration
    optimal_idx = grouped_data['completion_time']['mean'].idxmin()
    optimal_config = {
        'num_robots': float(grouped_data.loc[optimal_idx, 'num_robots']),
        'completion_time': float(grouped_data.loc[optimal_idx, ('completion_time', 'mean')]),
        'completion_time_std': float(grouped_data.loc[optimal_idx, ('completion_time', 'std')]),
        'orders_per_step': float(grouped_data.loc[optimal_idx, ('orders_per_step', 'mean')]),
        'orders_per_step_std': float(grouped_data.loc[optimal_idx, ('orders_per_step', 'std')])
    }
    
    # Add reference lines for optimal configuration
    ax1.axvline(x=optimal_config['num_robots'], color='g', linestyle='--', 
                label=f"Optimal: {optimal_config['num_robots']:.1f} robots")
    ax1.axhline(y=optimal_config['completion_time'], color='g', linestyle='--',
                label=f"Min time: {optimal_config['completion_time']:.0f} steps")
    
    ax1.set_xlabel('Number of Robots')
    ax1.set_ylabel('Completion Time (steps)')
    ax1.set_title(f'Completion Time vs Number of Robots\nWarehouse: 50.0x85.0, Orders: {target_orders}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add text box with optimal configuration details
    text = f'Optimal Configuration:\n'
    text += f'Robots: {optimal_config["num_robots"]:.1f}\n'
    text += f'Completion Time: {optimal_config["completion_time"]:.0f} ± {optimal_config["completion_time_std"]:.0f} steps\n'
    text += f'Orders/Step: {optimal_config["orders_per_step"]:.3f} ± {optimal_config["orders_per_step_std"]:.3f}'
    
    ax1.text(0.02, 0.98, text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Marginal Benefit
    ax2.bar(grouped_data['num_robots'][1:], grouped_data['marginal_benefit'][1:], 
            alpha=0.6, width=4, color='blue')
    ax2.set_xlabel('Number of Robots')
    ax2.set_ylabel('Time Steps Reduced\nper Additional 5 Robots')
    ax2.set_title('Marginal Benefit of Adding Robots')
    ax2.grid(True, alpha=0.3)
    
    # Add text box with marginal benefit insights
    significant_benefit_threshold = 50  # threshold for significant time reduction
    good_roi_robots = grouped_data[grouped_data['marginal_benefit'] > significant_benefit_threshold]['num_robots'].values
    if len(good_roi_robots) > 0:
        text = f'High Impact Regions:\n'
        text += f'Adding robots has highest impact\n'
        text += f'up to {good_roi_robots[-1]} robots\n'
        text += f'(>{significant_benefit_threshold} steps reduced per 5 robots)'
        
        ax2.text(0.98, 0.98, text,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set x-axis limits for both plots
    ax1.set_xlim(robot_range[0], robot_range[1])
    ax2.set_xlim(robot_range[0], robot_range[1])
    
    plt.tight_layout()
    plt.savefig('plots/robot_scaling_1000_orders_2ws_detailed.png', dpi=300)
    plt.close()
    
    # Print marginal benefit analysis
    print("\nMarginal Benefit Analysis:")
    print("Number of Robots | Time Reduction (per 5 robots) | Time Reduction (per robot)")
    print("-" * 70)
    for _, row in grouped_data[1:].iterrows():  # Skip first row as it has no marginal benefit
        num_robots = float(row['num_robots'])
        marginal_benefit = float(row['marginal_benefit'])
        marginal_benefit_per_robot = float(row['marginal_benefit_per_robot'])
        print(f"{num_robots:13.0f} | {marginal_benefit:26.1f} | {marginal_benefit_per_robot:23.1f}")
    
    # Calculate correlation and print summary
    print("\nOptimal configuration found:")
    print(f"Number of robots: {optimal_config['num_robots']:.1f}")
    print(f"Completion time: {optimal_config['completion_time']:.0f} ± {optimal_config['completion_time_std']:.0f} steps")
    print(f"Orders per step: {optimal_config['orders_per_step']:.3f} ± {optimal_config['orders_per_step_std']:.3f}")
    
    # Calculate correlation
    corr, _ = pearsonr(grouped_data['num_robots'], grouped_data['completion_time']['mean'])
    print(f"\nCorrelation between robots and completion time: {corr:.3f}")
    
    return filtered_data, optimal_config, corr

def analyze_optimal_robot_count(data):
    """Analyze optimal number of robots for different scenarios"""
    
    # Group by workstations and find configurations with minimum completion time
    optimal_configs = []
    
    for ws in sorted(data['num_workstations'].unique()):
        ws_data = data[data['num_workstations'] == ws]
        
        # Find optimal robot count for different order counts
        for orders in sorted(ws_data['total_orders'].unique()):
            orders_data = ws_data[ws_data['total_orders'] == orders]
            optimal_row = orders_data.loc[orders_data['completion_time'].idxmin()]
            
            optimal_configs.append({
                'num_workstations': ws,
                'total_orders': orders,
                'optimal_robots': optimal_row['num_robots'],
                'min_completion_time': optimal_row['completion_time'],
                'orders_per_step': optimal_row['orders_per_step']
            })
    
    optimal_df = pd.DataFrame(optimal_configs)
    
    # Plot optimal robot counts
    plt.figure(figsize=(12, 6))
    for ws in sorted(optimal_df['num_workstations'].unique()):
        ws_data = optimal_df[optimal_df['num_workstations'] == ws]
        plt.scatter(ws_data['total_orders'], ws_data['optimal_robots'], 
                   label=f'{ws} workstations', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(ws_data['total_orders'], ws_data['optimal_robots'], 1)
        p = np.poly1d(z)
        plt.plot(ws_data['total_orders'], p(ws_data['total_orders']), '--', alpha=0.5)
    
    plt.xlabel('Total Orders')
    plt.ylabel('Optimal Number of Robots')
    plt.title('Optimal Robot Count vs Order Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/optimal_robot_counts.png')
    plt.close()
    
    return optimal_df

def main():
    # Load data
    print("Loading training data...")
    data = pd.read_csv('warehouse_data_files/training_data.csv')
    
    # Analyze completion time vs robots for 1000 orders
    print("\nAnalyzing completion time vs number of robots for 1000 orders (2 workstations)...")
    filtered_data, optimal_config, correlation = analyze_completion_time_vs_robots(
        data, target_orders=1000, robot_range=(30, 150)
    )
    
    print("\nResults saved to:")
    print("- plots/robot_scaling_1000_orders_2ws_detailed.png")

if __name__ == "__main__":
    main() 