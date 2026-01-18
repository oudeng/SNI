"""
Graphical Abstract for SNI Paper
Statistical-Neural Interaction Networks for Interpretable Mixed-Type Data Imputation

Square format version with larger fonts - Improved alignment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Set up figure - more square format
fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)
ax.set_xlim(0, 8)
ax.set_ylim(0, 7.5)
ax.axis('off')

# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'

# Professional color scheme
BLUE = '#2E86AB'      # Statistical components
ORANGE = '#E85D04'    # Neural components  
GREEN = '#2D6A4F'     # Input/Data
CORAL = '#D62828'     # Outputs
GRAY = '#6C757D'      # Secondary text
DARK = '#343A40'      # Main text
WHITE = '#FFFFFF'

# ============================================================================
# TITLE (Top) - LARGER
# ============================================================================
ax.text(4, 7.15, 'Statistical-Neural Interaction (SNI)', 
        fontsize=18, fontweight='bold', ha='center', va='center', color='black')
ax.text(4, 6.75, 'Interpretable Mixed-Type Data Imputation', 
        fontsize=13, ha='center', va='center', color=GRAY)

# ============================================================================
# TOP ROW: INPUT (Left) and OUTPUT (Right)
# ============================================================================

# INPUT BOX (Top Left) - narrower to match content
input_box = FancyBboxPatch((0.3, 4.4), 3.0, 2.1,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=GREEN, alpha=0.08, 
                           edgecolor=GREEN, linewidth=2.5)
ax.add_patch(input_box)

# Input box center: 0.3 + 3.0/2 = 1.8
input_center_x = 1.8

ax.text(input_center_x, 6.25, 'Input', fontsize=14, fontweight='bold', ha='center', color=GREEN)

# Data matrix with missing values - CENTERED in input box
matrix_width = 4 * 0.5  # 4 columns * cell width
matrix_left = input_center_x - matrix_width / 2  # Center the matrix
matrix_bottom = 4.7
cell_w, cell_h = 0.5, 0.38
rows, cols = 4, 4

for i in range(rows):
    for j in range(cols):
        x = matrix_left + j * cell_w
        y = matrix_bottom + (rows - 1 - i) * cell_h
        
        is_missing = (i, j) in [(0, 2), (1, 1), (2, 3), (3, 0)]
        
        if is_missing:
            rect = Rectangle((x, y), cell_w-0.03, cell_h-0.03,
                            facecolor='#FFEBEE', edgecolor=CORAL, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + cell_w/2 - 0.015, y + cell_h/2 - 0.015, '?', 
                   fontsize=14, ha='center', va='center', 
                   color=CORAL, fontweight='bold')
        else:
            is_categorical = j >= 2
            cell_color = '#E3F2FD' if not is_categorical else '#FFF3E0'
            rect = Rectangle((x, y), cell_w-0.03, cell_h-0.03,
                            facecolor=cell_color, edgecolor=GRAY, linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)

# Legend - centered below matrix
legend_total_width = 2.5
legend_left = input_center_x - legend_total_width / 2
ax.add_patch(Rectangle((legend_left, 4.45), 0.2, 0.14, facecolor='#E3F2FD', edgecolor=GRAY, linewidth=0.5))
ax.text(legend_left + 0.28, 4.52, 'Continuous', fontsize=9, va='center', color=GRAY)
ax.add_patch(Rectangle((legend_left + 1.3, 4.45), 0.2, 0.14, facecolor='#FFF3E0', edgecolor=GRAY, linewidth=0.5))
ax.text(legend_left + 1.58, 4.52, 'Categorical', fontsize=9, va='center', color=GRAY)

# OUTPUT BOX (Top Right) - WIDER to contain all content
output_box_left = 3.8
output_box_width = 4.0
output_box = FancyBboxPatch((output_box_left, 4.4), output_box_width, 2.1,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=CORAL, alpha=0.08, 
                            edgecolor=CORAL, linewidth=2.5)
ax.add_patch(output_box)

# Output box center: 3.8 + 4.0/2 = 5.8
# Move elements left by ~0.25 (about 2 character widths)
output_elements_center_x = output_box_left + output_box_width / 2 - 0.25  # = 5.55

ax.text(output_box_left + output_box_width / 2, 6.25, 'Outputs', fontsize=14, fontweight='bold', ha='center', color=CORAL)

# ============================================================================
# Output elements - ALL CENTERED relative to output_elements_center_x
# Move everything up slightly (+0.08)
# ============================================================================
y_shift = 0.08  # shift up

# Define spacing: two upper elements symmetric around center
left_element_center = output_elements_center_x - 1.0   # Imputed Data center
right_element_center = output_elements_center_x + 1.0  # Dependency Network center

# Output 1: Imputed Data (left side)
ax.text(left_element_center, 5.9 + y_shift, 'Imputed Data', fontsize=11, fontweight='bold', ha='center', color=DARK)

# Matrix: 3 cols x 2 rows, cell size 0.28 x 0.2
imputed_matrix_width = 3 * 0.28
comp_left = left_element_center - imputed_matrix_width / 2
comp_bottom = 5.4 + y_shift
for i in range(2):
    for j in range(3):
        rect = Rectangle((comp_left + j*0.28, comp_bottom + (1-i)*0.2), 
                         0.26, 0.18, facecolor='#C8E6C9', edgecolor=GREEN, linewidth=0.5)
        ax.add_patch(rect)

# Output 2: Dependency Network (right side)
ax.text(right_element_center, 5.9 + y_shift, 'Dependency Network', fontsize=11, fontweight='bold', ha='center', color=DARK)

# Network nodes - centered around right_element_center
# Total width of network: about 1.0 (from leftmost to rightmost node center)
network_width = 1.0
network_left = right_element_center - network_width / 2
nodes = [
    (network_left, 5.55 + y_shift), 
    (network_left + 0.33, 5.75 + y_shift), 
    (network_left + 0.66, 5.55 + y_shift), 
    (network_left + 1.0, 5.75 + y_shift)
]
for x, y in nodes:
    circle = Circle((x, y), 0.1, facecolor=CORAL, alpha=0.6, edgecolor=CORAL)
    ax.add_patch(circle)

edges = [
    (nodes[0], nodes[1]), 
    (nodes[1], nodes[2]),
    (nodes[2], nodes[3]),
    (nodes[0], nodes[2])
]
for (x1, y1), (x2, y2) in edges:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=CORAL, lw=1.2, alpha=0.7))

# Output 3: Prior Coefficients (centered below, spanning full width)
ax.text(output_elements_center_x, 5.22 + y_shift, r'Prior Coefficients $\{\lambda_h\}$', fontsize=11, fontweight='bold',
        ha='center', color=DARK)

lambdas = [0.7, 0.3, 0.9, 0.5]
bar_bottom = 4.68 + y_shift
# Center the bars
max_bar_width = 1.2  # max lambda * 1.2
bar_left = output_elements_center_x - max_bar_width / 2
for i, lam in enumerate(lambdas):
    ax.barh(bar_bottom + i*0.12, lam*1.2, height=0.1, left=bar_left,
            color=CORAL, alpha=0.5 + lam*0.3, edgecolor=CORAL)

ax.text(output_elements_center_x, 4.52, '(interpretability)', fontsize=10, 
        ha='center', color=GRAY, style='italic')

# ============================================================================
# ARROWS: Input → SNI, SNI → Output
# ============================================================================
# Down arrow from Input
arrow_in = FancyArrowPatch((input_center_x, 4.35), (input_center_x, 3.9),
                           arrowstyle='->', mutation_scale=18,
                           linewidth=2.5, color=DARK)
ax.add_patch(arrow_in)

# Down arrow to Output (use box center, not elements center)
output_center_x = output_box_left + output_box_width / 2
arrow_out = FancyArrowPatch((output_center_x, 3.9), (output_center_x, 4.35),
                            arrowstyle='->', mutation_scale=18,
                            linewidth=2.5, color=DARK)
ax.add_patch(arrow_out)

# ============================================================================
# BOTTOM: SNI METHOD (Center, larger)
# ============================================================================
method_box = FancyBboxPatch((0.3, 0.65), 7.4, 3.05,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=WHITE, 
                            edgecolor=DARK, linewidth=2.5)
ax.add_patch(method_box)

ax.text(4, 3.45, 'SNI Framework', fontsize=15, fontweight='bold', ha='center', color='black')

# Statistical Step (Left)
stat_box = FancyBboxPatch((0.6, 1.35), 2.8, 1.9,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=BLUE, alpha=0.1, 
                          edgecolor=BLUE, linewidth=2)
ax.add_patch(stat_box)

ax.text(2.0, 3.0, 'Statistical Step', fontsize=13, fontweight='bold', ha='center', color=BLUE)
ax.text(2.0, 2.65, 'Correlation Prior', fontsize=11, ha='center', color=BLUE, style='italic')

# Correlation matrix icon
corr_left, corr_bottom = 1.3, 1.7
for i in range(3):
    for j in range(3):
        intensity = 1 - abs(i-j) * 0.3
        rect = Rectangle((corr_left + j*0.42, corr_bottom + (2-i)*0.22), 
                         0.4, 0.2, facecolor=BLUE, alpha=intensity*0.6)
        ax.add_patch(rect)

ax.text(2.0, 1.5, r'$\mathbf{P}_f$', fontsize=14, ha='center', color=BLUE, fontweight='bold')

# Neural Step (Right)
neural_box = FancyBboxPatch((4.6, 1.35), 2.8, 1.9,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=ORANGE, alpha=0.1, 
                            edgecolor=ORANGE, linewidth=2)
ax.add_patch(neural_box)

ax.text(6.0, 3.0, 'Neural Step', fontsize=13, fontweight='bold', ha='center', color=ORANGE)
ax.text(6.0, 2.65, 'Feature Attention', fontsize=11, ha='center', color=ORANGE, style='italic')

# Attention visualization
att_left, att_bottom = 5.2, 1.7
head_colors = [ORANGE, '#F4A261', '#E9C46A']
for h, hc in enumerate(head_colors):
    for i in range(3):
        alpha_val = [0.8, 0.4, 0.2][i] if h == 0 else [0.3, 0.7, 0.5][i] if h == 1 else [0.5, 0.3, 0.8][i]
        rect = Rectangle((att_left + h*0.5, att_bottom + (2-i)*0.22), 
                         0.48, 0.2, facecolor=hc, alpha=alpha_val)
        ax.add_patch(rect)

ax.text(6.0, 1.5, r'$\mathbf{A}^{(h)}$', fontsize=14, ha='center', color=ORANGE, fontweight='bold')

# EM iteration arrows (between stat and neural)
arrow1 = FancyArrowPatch((3.45, 2.45), (4.55, 2.45),
                         arrowstyle='->', mutation_scale=14,
                         linewidth=2, color=GRAY)
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((4.55, 2.15), (3.45, 2.15),
                         arrowstyle='->', mutation_scale=14,
                         linewidth=2, color=GRAY)
ax.add_patch(arrow2)

ax.text(4.0, 1.88, 'EM Iterations', fontsize=10, ha='center', va='center', color=GRAY)

# Key insight box (bottom of SNI) - CENTERED vertically and horizontally
eq_box_left = 1.1
eq_box_width = 5.8
eq_box_bottom = 0.78
eq_box_height = 0.45
eq_box = FancyBboxPatch((eq_box_left, eq_box_bottom), eq_box_width, eq_box_height,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor='#F8F9FA', edgecolor=GRAY, 
                        linewidth=1.2, linestyle='--')
ax.add_patch(eq_box)

# Text centered in the equation box
eq_text_x = eq_box_left + eq_box_width / 2
eq_text_y = eq_box_bottom + eq_box_height / 2
ax.text(eq_text_x, eq_text_y, r'Controllable Prior: $\lambda_h$ regularizes attention toward correlation prior',
        fontsize=11, ha='center', va='center', color=DARK)

# ============================================================================
# BOTTOM: KEY MESSAGE - LARGER FONT (+2)
# ============================================================================
ax.text(4, 0.25, 'Balances statistical rigor with neural flexibility for interpretable imputation',
        fontsize=13, ha='center', va='center', color=GRAY, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# ============================================================================
# Save
# ============================================================================
plt.tight_layout()
plt.savefig('graphical_abstract.png', dpi=300, 
            bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.savefig('graphical_abstract.pdf', 
            bbox_inches='tight', pad_inches=0.15, facecolor='white')
print("Improved graphical abstract saved!")
plt.close()