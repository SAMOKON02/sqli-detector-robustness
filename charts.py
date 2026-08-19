# Charts for the results chapter
import matplotlib.pyplot as plt
import numpy as np

# ---- Your actual results ----
detectors = ['Random Forest', 'LSTM']
clean = [99.0, 92.2]   # detection on clean attacks
before_adv = [38.8, 64.0]   # detection on ENCODED attacks (before hardening)
# detection on ENCODED attacks (after adversarial training)
after_adv = [99.6, 99.3]

x = np.arange(len(detectors))   # positions for the two detectors
width = 0.25                    # width of each bar

fig, ax = plt.subplots(figsize=(8, 5))

# three bars per detector: clean, before, after
bars1 = ax.bar(x - width, clean,      width,
               label='Clean attacks',            color='#4C72B0')
bars2 = ax.bar(x,         before_adv, width,
               label='Encoded (before hardening)', color='#C44E52')
bars3 = ax.bar(x + width, after_adv,  width,
               label='Encoded (after hardening)',  color='#55A868')

# labels and styling
ax.set_ylabel('Detection rate (%)')
ax.set_title(
    'Detection rate: clean vs encoded attacks, before and after adversarial training')
ax.set_xticks(x)
ax.set_xticklabels(detectors)
ax.set_ylim(0, 110)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3)

# put the number on top of each bar
for bars in (bars1, bars2, bars3):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('chart_detection_rates.png', dpi=200)   # saves the image
plt.show()                                          # opens it on screen
print("Chart saved as chart_detection_rates.png")
