''' 
Usage:
Create visualization of regression + CDF.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# create data for demonstration
x = np.array([1, 2, 3])
y = np.array([2, 3, 7])
x_line = np.array([1, 2, 3])
y_line = np.array([2, 4, 6])

# set up plot
fig, ax = plt.subplots(1, 1, figsize=(20,10))
FONTSIZE = 16

# plot data
ax.plot(np.linspace(0, 4), np.linspace(0, 4)*2, linewidth=4, alpha=0.5, label=r'$\hat{y} = f(x)$')
ax.scatter(x, y, color='darkblue', label=r'$(x_t, y_t)$')

i = 0
# iterate through demo points
for x_, y_, y_line_ in zip(x, y, y_line):
    # get y_t, H(x_t), [H(x_t)](y_t), 
    y_values = np.sort(np.append(np.linspace(y_line_-3, y_line_+3), [y_]))
    cdf_values = st.norm.cdf(y_values, loc=y_line_, scale=1)
    y_cdf_intersection = x_ - cdf_values[y_values == y_]
    
    # plot CDF at x_t and show where y_t meets CDF 
    if i ==0:
        ax.plot(-1*cdf_values + x_, y_values, color='darkred', alpha=0.6, label=r'$H(x_t)$')
        ax.plot([y_cdf_intersection, x_], [y_, y_], linestyle='dashed', color='darkred')
        ax.plot(y_cdf_intersection, y_, marker='o', markersize=10, color='darkred', label=r'$[H(x_t)](y_t)$')
        i += 1
    else:
        ax.plot(-1*cdf_values + x_, y_values, alpha=0.6, color='darkred')
        ax.plot([y_cdf_intersection, x_], [y_, y_], linestyle='dashed', color='darkred')
        ax.plot(y_cdf_intersection, y_, marker='o', markersize=10, color='darkred')

# legends, axes
ax.legend(loc='upper left', fontsize=FONTSIZE)
ax.set_xlabel('x', fontsize=FONTSIZE)
ax.set_ylabel('y', fontsize=FONTSIZE)
ax.set_title(r'Sketch of $[H(x_t)](y_t)$', fontsize=FONTSIZE+5)

plt.savefig('../images/regression_cdf_plot.png');