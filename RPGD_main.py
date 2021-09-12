import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from prettytable import PrettyTable


def run(sys1, sys2, sys3):

    # Pitches appear to already be sorted by Time but let's make sure
    sys1.sort_values(by='Time')
    sys2.sort_values(by='Time')
    sys3.sort_values(by='Time')

    # Rename columns so they are easier to keep track of when merging
    sys1 = sys1.rename(columns={"Time": "Time_1", "X": "X_1", "Y": "Y_1", "Z": "Z_1"})
    sys2 = sys2.rename(columns={"Time": "Time_2", "x": "X_2", "y": "Y_2", "z": "Z_2"})

    # Merge the DFs together using the nearest timestamp
    # (https://pandas.pydata.org/pandas-docs/version/0.25.0/reference/api/pandas.merge_asof.html)
    sys1_3 = pd.merge_asof(sys3, sys1, left_on='Time', right_on="Time_1", direction='nearest')
    sys2_3 = pd.merge_asof(sys3, sys2, left_on='Time', right_on="Time_2", direction='nearest')

    # Check to see how well the timestamps correlate to validate merging
    plt.figure(1, figsize=(12, 8))
    plt.plot(sys1_3['Time'], sys1_3['Time_1'], 'b.', label='System 1')
    plt.plot(sys2_3['Time'], sys2_3['Time_2'], 'r.', label='System 2')
    plt.xlabel('System 3')
    plt.ylabel('System 1 and System 2')
    plt.legend()
    plt.title('Timestamp Correlations to System 3')
    plt.show()

    # Plot each of the x,y,z points to get an idea how they data is matching up and which system is performing better
    # System 2 appears to have a significant advantage here
    plt.figure(2)
    plt.plot(sys1_3['X'], sys1_3['X_1'], 'b.', label='System 1')
    plt.plot(sys2_3['X'], sys2_3['X_2'], 'r.', label='System 2')
    plt.xlabel('System 3')
    plt.ylabel('System 1 and System 2')
    plt.legend()
    plt.title('X-axis Correlations to System 3')
    plt.show()

    # System 2 still looks better but the correlation to System 3 is not as strong
    plt.figure(3)
    plt.plot(sys1_3['Y'], sys1_3['Y_1'], 'b.', label='System 1')
    plt.plot(sys2_3['Y'], sys2_3['Y_2'], 'r.', label='System 2')
    plt.xlabel('System 3')
    plt.ylabel('System 1 and System 2')
    plt.legend()
    plt.title('Y-axis Correlations to System 3')
    plt.show()

    # System 2 is much better than System 1 for the z-axis also
    plt.figure(4)
    plt.plot(sys1_3['Z'], sys1_3['Z_1'], 'b.', label='System 1')
    plt.plot(sys2_3['Z'], sys2_3['Z_2'], 'r.', label='System 2')
    plt.xlabel('System 3')
    plt.ylabel('System 1 and System 2')
    plt.legend()
    plt.title('Z-axis Correlations to System 3')
    plt.show()

    # Calculate RMSE for each system and axis
    RMSE1_3_x = mean_squared_error(sys1_3['X'], sys1_3['X_1'])
    RMSE1_3_y = mean_squared_error(sys1_3['Y'], sys1_3['Y_1'])
    RMSE1_3_z = mean_squared_error(sys1_3['Z'], sys1_3['Z_1'])

    RMSE2_3_x = mean_squared_error(sys2_3['X'], sys2_3['X_2'])
    RMSE2_3_y = mean_squared_error(sys2_3['Y'], sys2_3['Y_2'])
    RMSE2_3_z = mean_squared_error(sys2_3['Z'], sys2_3['Z_2'])

    rmse = PrettyTable(float_format='.2')
    rmse.field_names = ["System ID", "X-axis", "Y-axis", "Z-axis"]
    rmse.add_rows(
        [
            ["System 1", round(RMSE1_3_x, 4), round(RMSE1_3_y, 4), round(RMSE1_3_z, 4)],
            ["System 2", round(RMSE2_3_x, 4), round(RMSE2_3_y, 4), round(RMSE2_3_z, 4)],
            ["Difference", round(RMSE1_3_x - RMSE2_3_x, 4), round(RMSE1_3_y - RMSE2_3_y, 4), round(RMSE1_3_z - RMSE2_3_z, 4)]
        ]
    )
    print(rmse)

    # Visualize system comparisons using a Bland-Altman plot (https://www.statsmodels.org/devel/generated/statsmodels.graphics.agreement.mean_diff_plot.html)
    f, ax = plt.subplots(2, figsize=(12, 8))
    sm.graphics.mean_diff_plot(sys1_3['X'], sys1_3['X_1'], ax=ax[0], scatter_kwds={'c': 'b', 'marker': '.'}, mean_line_kwds={'c': 'b'})
    sm.graphics.mean_diff_plot(sys2_3['X'], sys2_3['X_2'], ax=ax[1], scatter_kwds={'c': 'r', 'marker': '.'}, mean_line_kwds={'c': 'r'})
    plt.show()

    f, ax = plt.subplots(2, figsize=(12, 8))
    sm.graphics.mean_diff_plot(sys1_3['Y'], sys1_3['Y_1'], ax=ax[0], scatter_kwds={'c': 'b', 'marker': '.'}, mean_line_kwds={'c': 'b'})
    sm.graphics.mean_diff_plot(sys2_3['Y'], sys2_3['Y_2'], ax=ax[1], scatter_kwds={'c': 'r', 'marker': '.'}, mean_line_kwds={'c': 'r'})
    plt.show()

    f, ax = plt.subplots(2, figsize=(12, 8))
    sm.graphics.mean_diff_plot(sys1_3['Z'], sys1_3['Z_1'], ax=ax[0], scatter_kwds={'c': 'b', 'marker': '.'}, mean_line_kwds={'c': 'b'})
    sm.graphics.mean_diff_plot(sys2_3['Z'], sys2_3['Z_2'], ax=ax[1], scatter_kwds={'c': 'r', 'marker': '.'}, mean_line_kwds={'c': 'r'})
    plt.show()

    # Visualize each system in 2 of the 3 dimensions to look at clustering
    plt.figure(10)
    plt.plot(sys1_3['X_1'], sys1_3['Y_1'], 'b.', label='System 1')
    plt.plot(sys2_3['X_2'], sys2_3['Y_2'], 'r.', label='System 2')
    plt.plot(sys1_3['X'], sys1_3['Y'], 'g.', label='System 3')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('')
    plt.show()



    breakpoint()




if __name__ == "__main__":
    system1_raw = pd.read_csv('system1.csv', parse_dates=['Time'])
    system2_raw = pd.read_csv('system2.csv', parse_dates=['Time'])
    system3_raw = pd.read_csv('system3.csv', parse_dates=['Time'])
    run(system1_raw, system2_raw, system3_raw)