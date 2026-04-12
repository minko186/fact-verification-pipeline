import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. Set Global Aesthetic Parameters (for professional look) ---
# Use Seaborn's 'whitegrid' style for a clean background
sns.set_style("whitegrid")
# Configure font family and size for consistency across plots
# Common recommendations: Times New Roman, Computer Modern (LaTeX), or Sans-Serif
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.linestyle': '--', # Use dashed lines for grid
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5
})

def create_research_plots():
    # --- 2. Generate Sample Data ---
    X = np.linspace(0, 10, 50)
    # Data for the Line Plot (Subplot 1)
    Y1 = np.sin(X) * np.exp(-0.1 * X)
    Y2 = np.cos(X) * np.exp(-0.1 * X)
    # Data for the Bar Plot (Subplot 2)
    models = ['Model A', 'Model B', 'Model C', 'Model D']
    metrics = [0.85, 0.92, 0.78, 0.95]
    errors = [0.03, 0.02, 0.04, 0.01]

    # --- 3. Create Figure and Subplots ---
    # Use figsize appropriate for a journal column (e.g., 3.25in for single column)
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=2, 
        figsize=(7.0, 3.5), # Wide enough for two plots side-by-side
        constrained_layout=True # Automatically adjusts subplot params for tight layout
    )

    # --- Subplot 1: Time Series/Line Plot ---
    ax1 = axes[0]
    
    # Plotting multiple lines with clear colors
    ax1.plot(X, Y1, label=r'$f_1(x) = \sin(x)e^{-0.1x}$', color='tab:blue', linestyle='-')
    ax1.plot(X, Y2, label=r'$f_2(x) = \cos(x)e^{-0.1x}$', color='tab:red', linestyle='--')
    
    # Set clear labels and units
    ax1.set_xlabel('Time $(s)$')
    ax1.set_ylabel('Amplitude $(\mu V)$')
    
    # Set a descriptive legend
    ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    
    # Add subplot identifier for referencing in the paper
    ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # --- Subplot 2: Bar Plot with Error Bars ---
    ax2 = axes[1]
    
    # Create a bar plot, adding error bars which are crucial for research
    bars = ax2.bar(
        models, 
        metrics, 
        yerr=errors, 
        capsize=5, # Cap size for error bars
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Specific color mapping
    )

    # Add metric labels above bars for easy reading
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    # Set limits and ticks
    ax2.set_ylim(0.70, 1.0)
    ax2.set_xlabel('Model Architecture')
    # Use a descriptive label for the Y-axis metric
    ax2.set_ylabel('F1-Score (Mean $\pm$ Std. Dev.)')

    # Add subplot identifier
    ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    
    # --- 4. Finalizing and Saving the Figure ---
    # fig.suptitle('Figure 1: Experimental Results', fontsize=14, fontweight='bold') # Use sparingly or not at all

    # Save in high-resolution, vector format (required for most journals)
    # The 'dpi' and 'format' are critical here
    plt.savefig('research_quality_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Execute the function
create_research_plots()