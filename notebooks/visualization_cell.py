# ============================================================
# VISUALIZATION CELL - Add this to your notebook after Block 9
# Creates comparison plots matching your presentation requirements
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============ RESULTS TABLE ============
print("\n" + "="*60)
print("FINAL RESULTS - All Models")
print("="*60)

# Create results DataFrame
results_data = {
    'Model': ['Model A', 'Model B', 'Model C', 'Model D'],
    'Pretraining Source': ['ImageNet-1K', 'ImageNet-1K', 'Flowers-102', 'Flowers-102'],
    'Training Strategy': ['Linear Probing', 'Fine-Tuning', 'Linear Probing', 'Fine-Tuning'],
    'Backbone Status': ['Frozen', 'Trainable', 'Frozen', 'Trainable'],
    'Trainable Params': ['~5,130', '~11M', '~5,130', '~11M'],
    'Learning Rate': ['1e-3', '1e-4', '1e-3', '1e-4'],
    'Val Accuracy': [results.get('Model A', 0), results.get('Model B', 0),
                     results.get('Model C', 0), results.get('Model D', 0)]
}

df_results = pd.DataFrame(results_data)
df_results['Accuracy (%)'] = df_results['Val Accuracy'] * 100

print(df_results.to_string(index=False))

# Save to CSV
df_results.to_csv('../models/final_results_comparison.csv', index=False)
print(f"\nResults saved to: ../models/final_results_comparison.csv")

# ============ VISUALIZATION 1: Bar Chart Comparison ============
fig, ax = plt.subplots(figsize=(12, 6))

models = ['Model A\n(ImageNet\nLinear Probe)',
          'Model B\n(ImageNet\nFine-Tuning)',
          'Model C\n(Flowers\nLinear Probe)',
          'Model D\n(Flowers\nFine-Tuning)']

accuracies = [results.get('Model A', 0) * 100,
              results.get('Model B', 0) * 100,
              results.get('Model C', 0) * 100,
              results.get('Model D', 0) * 100]

colors = ['#3498DB', '#E67E22', '#27AE60', '#8E44AD']

bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparison of All Models: Pretraining Source × Training Strategy',
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('models_comparison_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nBar chart saved as: models_comparison_bar_chart.png")

# ============ VISUALIZATION 2: 2x2 Grid Comparison ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stage 1 Research Question: Pretraining Source × Training Strategy',
             fontsize=16, fontweight='bold', y=0.98)

model_configs = [
    ('Model A', 'ImageNet-1K', 'Linear Probing', 'Frozen', '#3498DB', results.get('Model A', 0)),
    ('Model B', 'ImageNet-1K', 'Fine-Tuning', 'Trainable', '#E67E22', results.get('Model B', 0)),
    ('Model C', 'Flowers-102', 'Linear Probing', 'Frozen', '#27AE60', results.get('Model C', 0)),
    ('Model D', 'Flowers-102', 'Fine-Tuning', 'Trainable', '#8E44AD', results.get('Model D', 0))
]

for idx, (model, pretrain, strategy, backbone, color, acc) in enumerate(model_configs):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Create pie chart showing accuracy vs remaining
    sizes = [acc * 100, 100 - acc * 100]
    colors_pie = [color, '#E0E0E0']
    explode = (0.1, 0)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_title(f'{model}\n{pretrain}\n{strategy} ({backbone})',
                 fontsize=12, fontweight='bold', pad=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('models_2x2_grid_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("2x2 grid comparison saved as: models_2x2_grid_comparison.png")

# ============ VISUALIZATION 3: Training History (if available) ============
if 'hist_A' in globals() and 'hist_B' in globals():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    epochs = range(1, len(hist_A['train_loss']) + 1)
    ax1.plot(epochs, hist_A['train_loss'], 'o-', color='#3498DB', label='Model A (Linear Probe)', linewidth=2)
    ax1.plot(epochs, hist_B['train_loss'], 's-', color='#E67E22', label='Model B (Fine-Tuning)', linewidth=2)

    if 'hist_C' in globals():
        ax1.plot(epochs, hist_C['train_loss'], '^-', color='#27AE60', label='Model C (Flower LP)', linewidth=2)
    if 'hist_D' in globals():
        ax1.plot(epochs, hist_D['train_loss'], 'd-', color='#8E44AD', label='Model D (Flower FT)', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, [x*100 for x in hist_A['val_acc']], 'o-', color='#3498DB', label='Model A', linewidth=2)
    ax2.plot(epochs, [x*100 for x in hist_B['val_acc']], 's-', color='#E67E22', label='Model B', linewidth=2)

    if 'hist_C' in globals():
        ax2.plot(epochs, [x*100 for x in hist_C['val_acc']], '^-', color='#27AE60', label='Model C', linewidth=2)
    if 'hist_D' in globals():
        ax2.plot(epochs, [x*100 for x in hist_D['val_acc']], 'd-', color='#8E44AD', label='Model D', linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Training history curves saved as: training_history_all_models.png")

# ============ KEY FINDINGS ============
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

# Compare Linear Probing vs Fine-Tuning
if results.get('Model A') and results.get('Model B'):
    imagenet_improvement = (results['Model B'] - results['Model A']) * 100
    print(f"\nImageNet Pretraining:")
    print(f"  Fine-Tuning vs Linear Probing: +{imagenet_improvement:.2f}% improvement")

if results.get('Model C') and results.get('Model D'):
    flower_improvement = (results['Model D'] - results['Model C']) * 100
    print(f"\nFlowers-102 Pretraining:")
    print(f"  Fine-Tuning vs Linear Probing: +{flower_improvement:.2f}% improvement")

# Compare Pretraining Sources
if results.get('Model A') and results.get('Model C'):
    lp_comparison = (results['Model C'] - results['Model A']) * 100
    print(f"\nLinear Probing Strategy:")
    print(f"  Flowers vs ImageNet: {'+' if lp_comparison > 0 else ''}{lp_comparison:.2f}%")

if results.get('Model B') and results.get('Model D'):
    ft_comparison = (results['Model D'] - results['Model B']) * 100
    print(f"\nFine-Tuning Strategy:")
    print(f"  Flowers vs ImageNet: {'+' if ft_comparison > 0 else ''}{ft_comparison:.2f}%")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
