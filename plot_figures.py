import matplotlib.pyplot as plt
import csv
import numpy as np
from tikzplotlib import save as tikz_save

# DEFAULT ACROBOT DATA
n_files = 3
default_values = []
default_values_dnn = []

for i in range(n_files):
    with open('../graph_csv/Acrobot_Default_' + str(i+1) + '.csv') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        data = []
        for row in plots:
            data.append(float(row[2]))
        default_values.append(data)

    with open('../graph_csv/Acrobot_Default_' + str(i+1) + '_DNN.csv') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        data = []
        for row in plots:
            data.append(float(row[2]))
        default_values_dnn.append(data)

default_values = np.asarray(default_values)
default_values_dnn = np.asarray(default_values_dnn)
mean_default_values = np.mean(default_values, 0)
mean_default_values_dnn = np.mean(default_values_dnn, 0)


# INTERPOLATION ACROBOT DATA
random_values = []
random_values_dnn = []

for i in range(n_files):
    with open('../graph_csv/Acrobot_Interpolation_' + str(i+1) + '.csv') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        data = []
        for row in plots:
            data.append(float(row[2]))
        random_values.append(data)

    with open('../graph_csv/Acrobot_Interpolation_' + str(i+1) + '_DNN.csv') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots)
        data = []
        for row in plots:
            data.append(float(row[2]))
        random_values_dnn.append(data)

random_values = np.asarray(random_values)
random_values_dnn = np.asarray(random_values_dnn)
mean_random_values = np.mean(random_values, 0)
mean_random_values_dnn = np.mean(random_values_dnn, 0)


time_step = np.arange(0, 20)
# default_values_1 = np.asarray(default_values_1)
# default_values_2 = np.asarray(default_values_2)
# default_values_3 = np.asarray(default_values_3)
# default_values = np.array([default_values_1, default_values_2, default_values_3])

# plt.plot(time_step, default_values_1, label='Default1')
# plt.plot(time_step, default_values_2, label='Default2')
# plt.plot(time_step, default_values_3, label='Default3')



ax = plt.plot(time_step, mean_default_values, 'b-', label='Default PNN')
plt.plot(time_step, mean_default_values_dnn, 'b--', label='Default DNN')
plt.plot(time_step, mean_random_values, 'g-', label='Random PNN')
plt.plot(time_step, mean_random_values_dnn, 'g--', label='Random DNN')
plt.axis([0.0, 20.0, -500.0, 0.0])
plt.yticks([0, -100, -200, -300, -400, -500])
plt.xlabel('Time Steps (in thousands)')
plt.ylabel('Avg. Reward')
plt.title('Acrobot Training Rewards')
plt.legend()
plt.grid(linestyle='--',)
plt.show()
tikz_save('figures/acrobot_rewards.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth', strict=True)
