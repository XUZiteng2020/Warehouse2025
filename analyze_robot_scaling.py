import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), height_ratios=[3, 2])
    plt.subplots_adjust(hspace=0.3)  # Increase space between subplots
    
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
                       label=f"{'Main Road' if color=='r' else 'Original'} Trend")
    
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
    
    # Adjust layout to prevent overlapping
    plt.gcf().set_size_inches(19, 14)  # Make figure wider to accommodate legends
    plt.tight_layout()
    plt.savefig('plots/robot_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print marginal benefit analysis
    print("\nMarginal Benefit Analysis:")
    print("\nMain Road Layout:")
    print("Number of Robots | Time Reduction (per 5 robots) | Time Reduction (per robot)")
    print("-" * 70)
    for _, row in grouped_main_road[1:].iterrows():
        print(f"{row['num_robots']:13.0f} | {row['marginal_benefit']:26.1f} | {row['marginal_benefit_per_robot']:23.1f}")
    
    print("\nOriginal Layout:")
    print("Number of Robots | Time Reduction (per 5 robots) | Time Reduction (per robot)")
    print("-" * 70)
    for _, row in grouped_original[1:].iterrows():
        print(f"{row['num_robots']:13.0f} | {row['marginal_benefit']:26.1f} | {row['marginal_benefit_per_robot']:23.1f}")
    
    # Calculate correlation and print summary
    corr_main, _ = pearsonr(grouped_main_road['num_robots'], grouped_main_road['completion_time_mean'])
    corr_orig, _ = pearsonr(grouped_original['num_robots'], grouped_original['completion_time_mean'])
    
    print("\nCorrelation between robots and completion time:")
    print(f"Main Road Layout: {corr_main:.3f}")
    print(f"Original Layout: {corr_orig:.3f}")
    
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

def load_layout_data(layout_file):
    """Load warehouse layout data with metadata"""
    metadata = {}
    layout_data = None
    
    with open(layout_file, 'r') as f:
        lines = f.readlines()
        # Read metadata from header comments
        for line in lines:
            if line.startswith('#'):
                key, value = line.strip('# \n').split(': ')
                metadata[key] = value
            else:
                break
        
        # Read layout data
        layout_data = np.loadtxt(layout_file, comments='#')
    
    return layout_data, metadata

def main():
    """Main analysis function"""
    # Load data
    print("Loading training data...")
    try:
        # Load layout data
        layout_file = 'layouts/warehouse_original.txt'
        layout_data, layout_metadata = load_layout_data(layout_file)
        
        # Load simulation data
        data_dir = 'layouts'  # Changed from warehouse_data_files
        data_main_road = pd.read_csv(os.path.join(data_dir, 'training_data_main_road.csv'))
        data_original = pd.read_csv(os.path.join(data_dir, 'training_data_original.csv'))
        
        print(f"Layout loaded: {layout_metadata['type']} ({layout_metadata['width']}x{layout_metadata['height']})")
    except FileNotFoundError as e:
        print(f"Error: Required data files not found in '{data_dir}' directory.")
        print("Please ensure the simulation data files are present in the layouts directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Set style for better visualization
    plt.style.use('seaborn')
    
    # Analyze completion time vs robots for 1000 orders
    print("\nAnalyzing completion time vs number of robots for 1000 orders (2 workstations)...")
    
    # Group and calculate statistics
    def prepare_data(df):
        # Group by number of robots
        grouped = df.groupby('num_robots').agg({
            'completion_time': ['mean', 'std'],
            'orders_per_step': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['num_robots'] + [f'{col[0]}_{col[1]}' for col in grouped.columns[1:]]
        
        # Sort by number of robots
        grouped = grouped.sort_values('num_robots')
        
        # Calculate marginal benefits
        grouped['marginal_benefit'] = -grouped['completion_time_mean'].diff()
        
        return grouped
    
    # Prepare data for both layouts
    grouped_main_road = prepare_data(data_main_road)
    grouped_original = prepare_data(data_original)
    
    # Create figure with adjusted size and spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 14), height_ratios=[3, 2])
    plt.subplots_adjust(hspace=0.3)
    
    # Plot 1: Completion Time vs Number of Robots
    def plot_completion_time(ax, main_road_data, original_data):
        # Plot main road data
        main_line = ax.errorbar(main_road_data['num_robots'], 
                              main_road_data['completion_time_mean'],
                              yerr=main_road_data['completion_time_std'],
                              fmt='ro-', linewidth=2, markersize=8, alpha=0.8,
                              label='Main Road Layout', capsize=5)
        
        # Plot original layout data
        orig_line = ax.errorbar(original_data['num_robots'], 
                              original_data['completion_time_mean'],
                              yerr=original_data['completion_time_std'],
                              fmt='bo-', linewidth=2, markersize=8, alpha=0.8,
                              label='Original Layout', capsize=5)
        
        # Add trend lines
        def add_trend_line(data, color):
            z = np.polyfit(data['num_robots'], data['completion_time_mean'], 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(data['num_robots'].min(), data['num_robots'].max(), 100)
            ax.plot(x_smooth, p(x_smooth), f'{color}--', alpha=0.5,
                   label=f"{'Main Road' if color=='r' else 'Original'} Trend")
        
        add_trend_line(main_road_data, 'r')
        add_trend_line(original_data, 'b')
        
        # Find and mark optimal points
        main_opt_idx = main_road_data['completion_time_mean'].idxmin()
        orig_opt_idx = original_data['completion_time_mean'].idxmin()
        
        main_opt = main_road_data.iloc[main_opt_idx]
        orig_opt = original_data.iloc[orig_opt_idx]
        
        # Add vertical lines for optimal points
        ax.axvline(x=main_opt['num_robots'], color='r', linestyle='--', alpha=0.3,
                  label=f"Optimal Main Road: {main_opt['num_robots']:.0f}")
        ax.axvline(x=orig_opt['num_robots'], color='b', linestyle='--', alpha=0.3,
                  label=f"Optimal Original: {orig_opt['num_robots']:.0f}")
        
        # Add text boxes with optimal configuration details
        text_main = (f'Main Road Layout Optimal:\n'
                    f'Robots: {main_opt["num_robots"]:.0f}\n'
                    f'Time: {main_opt["completion_time_mean"]:.0f} ± {main_opt["completion_time_std"]:.0f} steps\n'
                    f'Orders/Step: {main_opt["orders_per_step_mean"]:.3f} ± {main_opt["orders_per_step_std"]:.3f}')
        
        text_orig = (f'Original Layout Optimal:\n'
                    f'Robots: {orig_opt["num_robots"]:.0f}\n'
                    f'Time: {orig_opt["completion_time_mean"]:.0f} ± {orig_opt["completion_time_std"]:.0f} steps\n'
                    f'Orders/Step: {orig_opt["orders_per_step_mean"]:.3f} ± {orig_opt["orders_per_step_std"]:.3f}')
        
        # Add text boxes
        ax.text(1.02, 0.65, text_main, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.text(1.02, 0.35, text_orig, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Number of Robots')
        ax.set_ylabel('Completion Time (steps)')
        ax.set_title('Completion Time vs Number of Robots\nWarehouse: 50x85, Orders: 1000')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Plot 2: Marginal Benefits
    def plot_marginal_benefits(ax, main_road_data, original_data):
        width = 0.35  # Thinner bars
        x = np.arange(len(main_road_data['num_robots'][1:]))
        
        # Plot bars
        ax.bar(x - width/2, main_road_data['marginal_benefit'][1:],
               width, alpha=0.7, color='red', label='Main Road Layout',
               edgecolor='darkred', linewidth=1)
        ax.bar(x + width/2, original_data['marginal_benefit'][1:],
               width, alpha=0.7, color='blue', label='Original Layout',
               edgecolor='darkblue', linewidth=1)
        
        # Customize plot
        ax.set_xlabel('Number of Robots')
        ax.set_ylabel('Time Steps Reduced\nper Additional 5 Robots')
        ax.set_title('Marginal Benefit of Adding Robots')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(main_road_data['num_robots'][1:].astype(int), rotation=45)
        
        # Add high impact regions text
        threshold = 50
        text = 'High Impact Regions:\n'
        for data, name in [(main_road_data, 'Main Road'), (original_data, 'Original')]:
            high_impact = data[data['marginal_benefit'] > threshold]['num_robots'].values
            if len(high_impact) > 0:
                text += f'{name}: up to {high_impact[-1]:.0f} robots\n'
        text += f'(>{threshold} steps reduced per 5 robots)'
        
        ax.text(1.02, 0.65, text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create plots
    plot_completion_time(ax1, grouped_main_road, grouped_original)
    plot_marginal_benefits(ax2, grouped_main_road, grouped_original)
    
    # Final adjustments
    plt.tight_layout()
    plt.savefig('plots/robot_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis results
    print("\nResults saved to: plots/robot_scaling_comparison.png")

if __name__ == "__main__":
    main() 