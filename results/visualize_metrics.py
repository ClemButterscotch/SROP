import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_metrics(csv_filepath):
    """
    Loads metrics from a CSV file and generates bar charts for comparison.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        print("Please ensure the CSV file is in the same directory as this script, or provide the correct path.")
        return

    # Data preprocessing
    df['Wall-clock Time (s)'] = pd.to_numeric(df['Wall-clock Time (s)'])
    # Convert FLOPs from scientific notation string to numeric
    df['FLOPs'] = df['FLOPs'].apply(lambda x: float(x) if isinstance(x, (int, float)) else float(x.replace(',', '')))


    # Ensure 'Data' column is suitable for pivoting (e.g., 'Original', 'PCA')
    if not all(item in df['Data'].unique() for item in ['Original', 'PCA']):
        print("Error: 'Data' column must contain 'Original' and 'PCA' values for comparison.")
        return

    methods = df['Method'].unique()
    metrics_to_plot = {
        'Accuracy': 'Accuracy',
        'Wall-clock Time (s)': 'Wall-clock Time (s) (log scale)', # Using log scale for time
        'FLOPs': 'FLOPs (log scale)' # Using log scale for FLOPs
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(csv_filepath), 'charts')
    os.makedirs(output_dir, exist_ok=True)

    for metric_col, plot_title in metrics_to_plot.items():
        plt.figure(figsize=(12, 7))
        
        # Pivot table for easier plotting
        pivot_df = df.pivot(index='Method', columns='Data', values=metric_col)
        
        if pivot_df.empty or not all(col in pivot_df.columns for col in ['Original', 'PCA']):
             print(f"Warning: Could not create a valid pivot table for {metric_col}. Skipping this plot.")
             plt.close()
             continue

        # Reorder pivot_df rows to match methods order if necessary
        pivot_df = pivot_df.reindex(methods)

        num_methods = len(methods)
        bar_width = 0.35
        index = np.arange(num_methods)

        bar1 = plt.bar(index - bar_width/2, pivot_df['Original'], bar_width, label='Original Data')
        bar2 = plt.bar(index + bar_width/2, pivot_df['PCA'], bar_width, label='PCA Data')

        plt.xlabel('Method', fontsize=14)
        plt.ylabel(plot_title.split('(')[0].strip(), fontsize=14) # Get clean label
        plt.title(f'Comparison of {plot_title} With and Without PCA', fontsize=16)
        plt.xticks(index, methods, rotation=45, ha="right")
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if 'log scale' in plot_title:
            plt.yscale('log')
            # Add text annotations for log scale bars
            for bar_group in [bar1, bar2]:
                for bar in bar_group:
                    yval = bar.get_height()
                    if yval > 0: # Avoid log(0) issues if any bar is 0
                        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2e}', va='bottom', ha='center', fontsize=9)
        else:
            # Add text annotations for linear scale bars
            for bar_group in [bar1, bar2]:
                for bar in bar_group:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=9)
        
        chart_filename = f"{metric_col.replace(' ', '_').replace('(', '').replace(')', '')}_comparison.png"
        plt.savefig(os.path.join(output_dir, chart_filename))
        print(f"Saved chart: {os.path.join(output_dir, chart_filename)}")
        plt.show()

if __name__ == '__main__':
    # Assuming the CSV file is in the same directory as the script
    # If your CSV is in the parent directory (results/) and script is also in results/
    csv_file = 'MNIST-metrics-comparison.csv'
    
    # If the script is in 'results' and CSV is also in 'results'
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, csv_file)

    if not os.path.exists(csv_path):
        # Fallback: try to find it in the parent directory if script is in a subfolder of 'results'
        # Or if the script is in 'SROP' and csv is in 'SROP/results'
        parent_dir = os.path.dirname(script_dir) # SROP
        csv_path_alt = os.path.join(parent_dir, 'results', csv_file) # SROP/results/Fashion-MNIST-metrics-comparison.csv
        if os.path.exists(csv_path_alt):
            csv_path = csv_path_alt
        else: # Try SROP/Fashion-MNIST-metrics-comparison.csv if results folder doesn't exist
            csv_path_alt2 = os.path.join(parent_dir, csv_file)
            if os.path.exists(csv_path_alt2):
                 csv_path = csv_path_alt2
            else:
                # Final attempt: if script is in SROP/results, and CSV is in SROP (one level up from script_dir)
                # This case is less likely given the prompt, but good for robustness
                grandparent_dir = os.path.dirname(parent_dir) # Potentially clustering
                csv_path_alt3 = os.path.join(parent_dir, csv_file) # SROP/Fashion-MNIST-metrics-comparison.csv
                if os.path.exists(csv_path_alt3):
                    csv_path = csv_path_alt3


    visualize_metrics(csv_path)
