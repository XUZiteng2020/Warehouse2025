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

def analyze_completion_time_vs_robots(data_main_road: pd.DataFrame, data_original: pd.DataFrame, 
                                target_orders=1000, robot_range=(30, 150)):
    """Analyze relationship between completion time and number of robots for specific order count"""
    
    # Filter data for target order count (allow some variation) and 2 workstations
    order_margin = 0.1  # Allow 10% variation in order count
    
    def filter_data(data):
        filtered = data[
            (data['num_robots'] >= robot_range[0]) &
            (data['num_robots'] <= robot_range[1]) &
            (data['num_workstations'] == 2)  # Only 2 workstations
        ].copy()
        return filtered
    
    filtered_main_road = filter_data(data_main_road)
    filtered_original = filter_data(data_original)
    
    # Calculate mean and std for each robot count
    def group_data(data):
        grouped = data.groupby('num_robots').agg({
            'completion_time': ['mean', 'std'],
            'orders_per_step': ['mean', 'std']
        })
        # Flatten column names
        grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
        grouped = grouped.reset_index()
        return grouped
    
    grouped_main_road = group_data(filtered_main_road)
    grouped_original = group_data(filtered_original)
    
    # Sort by number of robots to ensure proper trend line calculation
    grouped_main_road = grouped_main_road.sort_values('num_robots')
    grouped_original = grouped_original.sort_values('num_robots')
    
    # Calculate marginal benefit for both layouts
    for df in [grouped_main_road, grouped_original]:
        df['marginal_benefit'] = -(df['completion_time_mean'].diff())  # per 5 robots
        df['marginal_benefit_per_robot'] = df['marginal_benefit'] / 5  # per single robot
    
    # Create visualization with more space
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18), height_ratios=[3, 2, 2])
    plt.subplots_adjust(hspace=0.4)  # Increased space between subplots
    
    # Plot 1: Completion Time vs Number of Robots
    # Main road layout
    ax1.errorbar(grouped_main_road['num_robots'], 
                grouped_main_road['completion_time_mean'],
                yerr=grouped_main_road['completion_time_std'],
                fmt='ro-', linewidth=2, markersize=8, alpha=0.8,
                label='Main Road Layout', capsize=5)
    
    # Original layout
    ax1.errorbar(grouped_original['num_robots'], 
                grouped_original['completion_time_mean'],
                yerr=grouped_original['completion_time_std'],
                fmt='bo-', linewidth=2, markersize=8, alpha=0.8,
                label='Original Layout', capsize=5)
    
    # Add trend lines
    def add_trend_line(data, color, alpha=0.5):
        z = np.polyfit(data['num_robots'], data['completion_time_mean'], 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(data['num_robots'].min(), data['num_robots'].max(), 100)
        return ax1.plot(x_smooth, p(x_smooth), f'{color}--', alpha=alpha, 
                       label=f'{"Main Road" if color=="r" else "Original"} Trend')
    
    add_trend_line(grouped_main_road, 'r')
    add_trend_line(grouped_original, 'b')
    
    # Find optimal configurations
    def find_optimal(data):
        optimal_idx = data['completion_time_mean'].idxmin()
        return {
            'num_robots': data.loc[optimal_idx, 'num_robots'],
            'completion_time': data.loc[optimal_idx, 'completion_time_mean'],
            'completion_time_std': data.loc[optimal_idx, 'completion_time_std'],
            'orders_per_step': data.loc[optimal_idx, 'orders_per_step_mean'],
            'orders_per_step_std': data.loc[optimal_idx, 'orders_per_step_std']
        }
    
    optimal_main_road = find_optimal(grouped_main_road)
    optimal_original = find_optimal(grouped_original)
    
    # Add reference lines for optimal configurations
    ax1.axvline(x=optimal_main_road['num_robots'], color='r', linestyle='--', alpha=0.3,
                label=f"Optimal Main Road: {optimal_main_road['num_robots']:.1f}")
    ax1.axvline(x=optimal_original['num_robots'], color='b', linestyle='--', alpha=0.3,
                label=f"Optimal Original: {optimal_original['num_robots']:.1f}")
    
    ax1.set_xlabel('Number of Robots')
    ax1.set_ylabel('Completion Time (steps)')
    ax1.set_title(f'Completion Time vs Number of Robots\nWarehouse: 50.0x85.0, Orders: {target_orders}')
    
    # Move legend outside the plot
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax1.grid(True, alpha=0.3)
    
    # Add text boxes with optimal configuration details
    text_main = f'Main Road Layout Optimal:\n'
    text_main += f'Robots: {optimal_main_road["num_robots"]:.1f}\n'
    text_main += f'Time: {optimal_main_road["completion_time"]:.0f} ± {optimal_main_road["completion_time_std"]:.0f} steps\n'
    text_main += f'Orders/Step: {optimal_main_road["orders_per_step"]:.3f} ± {optimal_main_road["orders_per_step_std"]:.3f}'
    
    text_original = f'Original Layout Optimal:\n'
    text_original += f'Robots: {optimal_original["num_robots"]:.1f}\n'
    text_original += f'Time: {optimal_original["completion_time"]:.0f} ± {optimal_original["completion_time_std"]:.0f} steps\n'
    text_original += f'Orders/Step: {optimal_original["orders_per_step"]:.3f} ± {optimal_original["orders_per_step_std"]:.3f}'
    
    # Place text boxes on the right side of the plot
    ax1.text(1.02, 0.65, text_main,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
             
    ax1.text(1.02, 0.35, text_original,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: Marginal Benefit
    width = 0.8  # Thinner bars (reduced from 1.6)
    x = np.arange(len(grouped_main_road['num_robots'][1:]))
    
    # Main road layout bars with adjusted position
    ax2.bar(x - width/2, grouped_main_road['marginal_benefit'][1:], 
            alpha=0.7, width=width, color='red', label='Main Road Layout',
            edgecolor='darkred', linewidth=1)
    
    # Original layout bars with adjusted position
    ax2.bar(x + width/2, grouped_original['marginal_benefit'][1:], 
            alpha=0.7, width=width, color='blue', label='Original Layout',
            edgecolor='darkblue', linewidth=1)
    
    ax2.set_xlabel('Number of Robots')
    ax2.set_ylabel('Time Steps Reduced\nper Additional 5 Robots')
    ax2.set_title('Marginal Benefit of Adding Robots')
    ax2.grid(True, alpha=0.3)
    
    # Move legend outside the plot
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Set x-ticks to show robot counts with adjusted spacing
    ax2.set_xticks(x)
    ax2.set_xticklabels(grouped_main_road['num_robots'][1:].astype(int), rotation=45)
    
    # Ensure bars are centered and spaced properly
    ax2.set_xlim(-1, len(x))
    
    # Add text box with marginal benefit insights
    significant_benefit_threshold = 50
    
    def get_high_impact_text(data, layout_name):
        good_roi_robots = data[data['marginal_benefit'] > significant_benefit_threshold]['num_robots'].values
        if len(good_roi_robots) > 0:
            return f'{layout_name}:\nHigh impact up to {good_roi_robots[-1]} robots\n'
        return ''
    
    text = 'High Impact Regions:\n'
    text += get_high_impact_text(grouped_main_road, 'Main Road')
    text += get_high_impact_text(grouped_original, 'Original')
    text += f'(>{significant_benefit_threshold} steps reduced per 5 robots)'
    
    # Place text box on the right side
    ax2.text(1.02, 0.65, text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set x-axis limits for both plots
    ax1.set_xlim(robot_range[0], robot_range[1])
    ax2.set_xlim(-1, len(x))
    
    # Plot 3: Percentage Improvement Histogram
    # Calculate percentage improvement
    improvement_data = pd.DataFrame({
        'num_robots': grouped_main_road['num_robots'],
        'main_road_time': grouped_main_road['completion_time_mean'],
        'original_time': grouped_original['completion_time_mean']
    })
    
    improvement_data['percent_improvement'] = ((improvement_data['original_time'] - improvement_data['main_road_time']) / 
                                             improvement_data['original_time'] * 100)
    
    # Create bar plot for percentage improvement
    bars = ax3.bar(improvement_data['num_robots'], improvement_data['percent_improvement'],
                  color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    
    # Add a horizontal line at 0%
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top')
    
    ax3.set_xlabel('Number of Robots')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Percentage Improvement of Main Road Layout vs Original Layout')
    ax3.grid(True, alpha=0.3)
    
    # Set x-axis limits
    ax3.set_xlim(robot_range[0], robot_range[1])
    
    # Add text box with summary statistics
    mean_improvement = improvement_data['percent_improvement'].mean()
    max_improvement = improvement_data['percent_improvement'].max()
    max_improvement_robots = improvement_data.loc[improvement_data['percent_improvement'].idxmax(), 'num_robots']
    
    summary_text = f'Summary Statistics:\n'
    summary_text += f'Mean Improvement: {mean_improvement:.1f}%\n'
    summary_text += f'Max Improvement: {max_improvement:.1f}%\n'
    summary_text += f'Best at {max_improvement_robots} robots'
    
    ax3.text(1.02, 0.65, summary_text,
             transform=ax3.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Adjust layout to prevent overlapping
    plt.gcf().set_size_inches(19, 18)  # Increased height to accommodate new subplot
    plt.tight_layout()
    plt.savefig('plots/robot_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print improvement analysis
    print("\nPercentage Improvement Analysis:")
    print("Number of Robots | Improvement (%)")
    print("-" * 35)
    for _, row in improvement_data.iterrows():
        print(f"{row['num_robots']:13.0f} | {row['percent_improvement']:15.1f}")
    
    print(f"\nMean improvement across all configurations: {mean_improvement:.1f}%")
    print(f"Maximum improvement: {max_improvement:.1f}% (at {max_improvement_robots} robots)")
    
    return filtered_main_road, filtered_original, optimal_main_road, optimal_original

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
    data_main_road = pd.read_csv('warehouse_data_files/training_data_main_road.csv')
    data_original = pd.read_csv('warehouse_data_files/training_data_original.csv')
    
    # Ensure numeric types
    data_main_road['num_robots'] = pd.to_numeric(data_main_road['num_robots'])
    data_main_road['completion_time'] = pd.to_numeric(data_main_road['completion_time'])
    data_original['num_robots'] = pd.to_numeric(data_original['num_robots'])
    data_original['completion_time'] = pd.to_numeric(data_original['completion_time'])
    
    # Analyze completion time vs robots for 1000 orders
    print("\nAnalyzing completion time vs number of robots for 1000 orders (2 workstations)...")
    filtered_main_road, filtered_original, optimal_main_road, optimal_original = analyze_completion_time_vs_robots(
        data_main_road, data_original, target_orders=1000, robot_range=(30, 150)
    )
    
    print("\nResults saved to:")
    print("- plots/robot_scaling_comparison.png")

if __name__ == "__main__":
    main() 