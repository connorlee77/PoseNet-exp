import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

DATA_DIR = 'predictions/'

# yp_king = np.load(DATA_DIR + 'position_king.npy')
# yq_king = np.load(DATA_DIR + 'orientation_king.npy')
# metrics_king = np.load(DATA_DIR + 'orientation_king_metrics.npy')

# TRUE_DIR = 'KingsCollege/'
# TEST_Y = 'kings_test_y.npy'

# f, axarr = plt.subplots(2, 2)


# ### Data gathering & preprocessing
# y_test = np.float32(np.load(TRUE_DIR + TEST_Y))
# y_test_x = y_test[:,0:3]
# y_test_q = y_test[:,3:]

# axarr[0, 0].scatter(y_test_x[:,0], y_test_x[:,1], s=15, marker='x', label='Ground Truth')
# axarr[0, 0].scatter(yp_king[:,0], yp_king[:,1], s=10, label='Predicted Position')
# axarr[0, 0].set_title('Kings College')
# axarr[0, 0].set_ylabel('y')
# yp_hospital = np.load(DATA_DIR + 'position_hospital.npy')
# yq_hospital = np.load(DATA_DIR + 'orientation_hospital.npy')
# metrics_hospital = np.load(DATA_DIR + 'orientation_hospital_metrics.npy')

# yp_hospital2 = np.load(DATA_DIR + 'position_hospital2.npy')
# yq_hospital2 = np.load(DATA_DIR + 'orientation_hospital2.npy')
# metrics_hospital2 = np.load(DATA_DIR + 'orientation_hospital_metrics2.npy')

# yp_hospital4 = np.load(DATA_DIR + 'position_hospital4.npy')
# yq_hospital4 = np.load(DATA_DIR + 'orientation_hospital4.npy')
# metrics_hospital4 = np.load(DATA_DIR + 'orientation_hospital_metrics4.npy')

# TRUE_DIR = 'OldHospital/'
# TEST_Y = 'hospital_test_y.npy'

# y_test = np.float32(np.load(TRUE_DIR + TEST_Y))
# y_test_x = y_test[:,0:3]
# y_test_q = y_test[:,3:]



# axarr[0, 1].scatter(y_test_x[:,0], y_test_x[:,1], s=15, marker='x', label='Ground Truth')
# axarr[0, 1].scatter(yp_hospital[:,0], yp_hospital[:,1], s=10, label='Predicted Position')
# axarr[0, 1].set_title('Old Hospital')

# axarr[1, 0].scatter(y_test_x[:,0], y_test_x[:,1], s=15, marker='x', label='Ground Truth')
# axarr[1, 0].scatter(yp_hospital2[:,0], yp_hospital2[:,1], s=10, label='Predicted Position')
# axarr[1, 0].set_title('Old Hospital (50% of Training Data)')
# axarr[1, 0].set_xlabel('x')
# axarr[1, 0].set_ylabel('y')

# axarr[1, 1].scatter(y_test_x[:,0], y_test_x[:,1], s=15, marker='x', label='Ground Truth')
# axarr[1, 1].scatter(yp_hospital4[:,0], yp_hospital4[:,1], s=10, label='Predicted Position')
# axarr[1, 1].set_title('Old Hospital (25% of Training Data)')
# axarr[1, 1].set_xlabel('x')
# plt.legend(loc='center', bbox_to_anchor=(-0.1, -0.20),
#           fancybox=True, ncol=5)
# plt.show()

f, axarr = plt.subplots(2, 3)

TRUE_DIR = 'KingsCollege/'
DATA_DP = 'y_labels_test_x.npy'
n = 200
y_data_dp = np.float32(np.load(DATA_DIR + DATA_DP))
cum_y_data_dp = np.cumsum(y_data_dp, axis=0)

yp_king = np.load(DATA_DIR + 'position_king_slam.npy')
vel = yp_king 
yp_king = np.cumsum(yp_king, axis=0)

axarr[0, 0].scatter(cum_y_data_dp[:n][:,0], cum_y_data_dp[:n][:,1], s=15, marker='x', label='Ground Truth')
axarr[0, 0].scatter(yp_king[:n][:,0], yp_king[:n][:,1], s=10, label='Predicted Position')
axarr[0, 0].set_title('After ' + str(n) + ' steps')
axarr[0, 0].set_xlabel('x')
axarr[0, 0].set_ylabel('y')
axarr[0, 0].legend()

n = 400
axarr[0, 1].scatter(cum_y_data_dp[:n][:,0], cum_y_data_dp[:n][:,1], s=15, marker='x', label='Ground Truth')
axarr[0, 1].scatter(yp_king[:n][:,0], yp_king[:n][:,1], s=10, label='Predicted Position')
axarr[0, 1].set_title('After ' + str(n) + ' steps)')
axarr[0, 1].set_xlabel('x')
axarr[0, 1].set_ylabel('y')

n = 600
axarr[0, 2].scatter(cum_y_data_dp[:n][:,0], cum_y_data_dp[:n][:,1], s=15, marker='x', label='Ground Truth')
axarr[0, 2].scatter(yp_king[:n][:,0], yp_king[:n][:,1], s=10, label='Predicted Position')
axarr[0, 2].set_title('After ' + str(n) + ' steps)')
axarr[0, 2].set_xlabel('x')
axarr[0, 2].set_ylabel('y')

n = 800
axarr[1, 0].scatter(cum_y_data_dp[:n][:,0], cum_y_data_dp[:n][:,1], s=15, marker='x', label='Ground Truth')
axarr[1, 0].scatter(yp_king[:n][:,0], yp_king[:n][:,1], s=10, label='Predicted Position')
axarr[1, 0].set_title('After ' + str(n) + ' steps)')
axarr[1, 0].set_xlabel('x')
axarr[1, 0].set_ylabel('y')

axarr[1, 1].plot(np.cumsum(np.abs(cum_y_data_dp[:n] - yp_king[:n]), axis=0))
axarr[1, 1].set_xlabel('Step')
axarr[1, 1].set_ylabel('Cumulative Error (m)')
axarr[1, 1].set_title('Accumulated Position Error')

axarr[1, 2].plot(np.abs(y_data_dp - vel))
axarr[1, 2].set_xlabel('Step')
axarr[1, 2].set_ylabel('Error (m/step)')
axarr[1, 2].set_title('Velocity Error')
axarr[1, 2].legend(['x', 'y', 'z'])

plt.suptitle('Kings College Visual Odometry')

plt.show()
