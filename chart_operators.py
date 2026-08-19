# Per-operator degradation chart
import matplotlib.pyplot as plt
import numpy as np

# ---- Your actual evasion-harness results ----
operators = ['Baseline\n(no disguise)', 'Comment',
             'Case', 'Whitespace', 'Encoding']
# Random Forest detection rate per operator
rf = [99.0, 98.2, 99.0, 99.0, 38.6]
lstm = [92.2, 92.0, 92.2, 94.0, 22.0]   # LSTM detection rate per operator

x = np.arange(len(operators))   # one group per operator
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, rf,   width,
               label='Random Forest', color='#4C72B0')
bars2 = ax.bar(x + width/2, lstm, width,
               label='LSTM',          color='#DD8452')

ax.set_ylabel('Detection rate (%)')
ax.set_title(
    'Detection rate by obfuscation operator (per-operator degradation)')
ax.set_xticks(x)
ax.set_xticklabels(operators)
ax.set_ylim(0, 110)
ax.legend()

# put the number on top of each bar
for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('chart_operators.png', dpi=200)
plt.show()
print("Chart saved as chart_operators.png")
