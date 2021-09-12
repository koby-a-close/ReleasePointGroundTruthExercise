import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from prettytable import PrettyTable


def run(src1, src2, src3):

    # Pitches appear to already be sorted by Time but let's make sure
    src1.sort_values(by='Time')
    src2.sort_values(by='Time')
    src3.sort_values(by='Time')

    # Rename columns so they are easier to keep track of when merging
    src1 = src1.rename(columns={"Time": "Time_1", "X": "X_1", "Y": "Y_1", "Z": "Z_1"})
    src2 = src2.rename(columns={"Time": "Time_2", "x": "X_2", "y": "Y_2", "z": "Z_2"})

    # Merge the DFs together using the nearest timestamp
    # (https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.merge_asof.html)
    src1_3 = pd.merge_asof(src3, src1, left_on='Time', right_on="Time_1", direction='nearest')
    src2_3 = pd.merge_asof(src3, src2, left_on='Time', right_on="Time_2", direction='nearest')

    # Check to see how well the timestamps line up to visually identify any bad matches
    plt.figure(1, figsize=(12, 8))
    plt.plot(src1_3['Time'], src1_3['Time_1'], 'b.', label='Source 1')
    plt.plot(src2_3['Time'], src2_3['Time_2'], 'r.', label='Source 2')
    plt.xlabel('Source 3')
    plt.ylabel('Source 1 and Source 2')
    plt.legend()
    plt.title('Timestamp Correlations to Source 3')
    plt.show()

    # Plot each of the x,y,z points to get an idea how they data is matching up and which Source is performing better
    # Source 2 appears to have a significant advantage here
    plt.figure(2)
    plt.plot(src1_3['X'], src1_3['X_1'], 'b.', label='Source 1')
    plt.plot(src2_3['X'], src2_3['X_2'], 'r.', label='Source 2')
    plt.xlabel('Source 3')
    plt.ylabel('Source 1 and Source 2')
    plt.legend()
    plt.title('X-axis Correlations to Source 3')
    plt.show()

    # Source 2 still looks better but the correlation to Source 3 is not as strong
    plt.figure(3)
    plt.plot(src1_3['Y'], src1_3['Y_1'], 'b.', label='Source 1')
    plt.plot(src2_3['Y'], src2_3['Y_2'], 'r.', label='Source 2')
    plt.xlabel('Source 3')
    plt.ylabel('Source 1 and Source 2')
    plt.legend()
    plt.title('Y-axis Correlations to Source 3')
    plt.show()

    # Source 2 is much better than Source 1 for the z-axis also
    plt.figure(4)
    plt.plot(src1_3['Z'], src1_3['Z_1'], 'b.', label='Source 1')
    plt.plot(src2_3['Z'], src2_3['Z_2'], 'r.', label='Source 2')
    plt.xlabel('Source 3')
    plt.ylabel('Source 1 and Source 2')
    plt.legend()
    plt.title('Z-axis Correlations to Source 3')
    plt.show()

    # Calculate MAE for each Source and axis
    MAE1_3_x = mean_absolute_error(src1_3['X'], src1_3['X_1'])
    MAE1_3_y = mean_absolute_error(src1_3['Y'], src1_3['Y_1'])
    MAE1_3_z = mean_absolute_error(src1_3['Z'], src1_3['Z_1'])
    MAE2_3_x = mean_absolute_error(src2_3['X'], src2_3['X_2'])
    MAE2_3_y = mean_absolute_error(src2_3['Y'], src2_3['Y_2'])
    MAE2_3_z = mean_absolute_error(src2_3['Z'], src2_3['Z_2'])

    # Take the MAE values and create a simple table to compare values. Source 3 uses the mean of its reported error
    MAE = PrettyTable()
    MAE.field_names = ["Source ID", "X-axis", "Y-axis", "Z-axis"]
    MAE.add_rows(
        [
            ["Source 1", round(MAE1_3_x, 4), round(MAE1_3_y, 4), round(MAE1_3_z, 4)],
            ["Source 2", round(MAE2_3_x, 4), round(MAE2_3_y, 4), round(MAE2_3_z, 4)],
            ["Source 3", round(np.mean(src3['Xrange']), 4), round(np.mean(src3['Yrange']), 4), round(np.mean(src3['Zrange']), 4)],
        ]
    )
    print(MAE)

    # Visualize Source comparisons using a Bland-Altman plot
    # To make things easier to pair I put the two sources as subplots, grouped by the axis
    # (https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html)
    f, ax = plt.subplots(2,1, figsize=(12,8))
    sm.graphics.mean_diff_plot(src1_3['X'], src1_3['X_1'], ax=ax[0], scatter_kwds={'c': 'b', 'marker': '.'}, mean_line_kwds={'c': 'b'})
    sm.graphics.mean_diff_plot(src2_3['X'], src2_3['X_2'], ax=ax[1], scatter_kwds={'c': 'r', 'marker': '.'}, mean_line_kwds={'c': 'r'})
    f.suptitle('Bland-Alman plot for the X-axis', va='top')
    ax[0].set_title('Source 1')
    ax[1].set_title('Source 2')
    plt.tight_layout()
    plt.show()

    f, ax = plt.subplots(2,1, figsize=(12, 8))
    sm.graphics.mean_diff_plot(src1_3['Y'], src1_3['Y_1'], ax=ax[0], scatter_kwds={'c': 'b', 'marker': '.'}, mean_line_kwds={'c': 'b'})
    sm.graphics.mean_diff_plot(src2_3['Y'], src2_3['Y_2'], ax=ax[1], scatter_kwds={'c': 'r', 'marker': '.'}, mean_line_kwds={'c': 'r'})
    f.suptitle('Bland-Alman plot for the Y-axis')
    ax[0].set_title('Source 1')
    ax[1].set_title('Source 2')
    plt.tight_layout()
    plt.show()

    f, ax = plt.subplots(2,1, figsize=(12, 8))
    sm.graphics.mean_diff_plot(src1_3['Z'], src1_3['Z_1'], ax=ax[0], scatter_kwds={'c': 'b', 'marker': '.'}, mean_line_kwds={'c': 'b'})
    sm.graphics.mean_diff_plot(src2_3['Z'], src2_3['Z_2'], ax=ax[1], scatter_kwds={'c': 'r', 'marker': '.'}, mean_line_kwds={'c': 'r'})
    f.suptitle('Bland-Alman plot for the Z-axis')
    ax[0].set_title('Source 1')
    ax[1].set_title('Source 2')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # I saved the data as individual csv files and load them here before passing into the main processing
    source1_raw = pd.read_csv('system1.csv', parse_dates=['Time'])
    source2_raw = pd.read_csv('system2.csv', parse_dates=['Time'])
    source3_raw = pd.read_csv('system3.csv', parse_dates=['Time'])
    run(source1_raw, source2_raw, source3_raw)