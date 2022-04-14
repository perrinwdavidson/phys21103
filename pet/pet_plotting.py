# ---------------------------------------------------------
# pet_reconstruct
# ---------------------------------------------------------
# import packages -----------------------------------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.transform import iradon

# load data -----------------------------------------------
# set number of sources ::
src_num = 3

# set filename and angle step (need to update accordingly) ::
if src_num == 1:
    data_filename = 'io/inputs/sinogram/Sinogram_Apr14_Time_19_32_17.csv'
    angle_step = 45
    dist_step = 1.0
    high_val = 6E2
elif src_num == 2:
    data_filename = 'io/inputs/sinogram/Sinogram_Apr07_Time_20_24_32.csv'
    angle_step = 45
    dist_step = 1.0
    high_val = 6E3
else:
    data_filename = 'io/inputs/sinogram/Sinogram_Apr07_Time_21_31_05.csv'
    angle_step = 5
    dist_step = 1.0
    high_val = 5E3

# load raw data ::
data = pd.read_csv(data_filename, delimiter=',', header=None)

# get meshgrid ::
sino = data.iloc[12:, :].to_numpy()  # 12: skipping header content
sino = sino.astype(float)

# get angles ::
angles = []
for i in range(np.shape(sino)[0]):
    angles.append(i * angle_step)

# get distances ::
dists = []
for i in range(np.shape(sino)[1]):
    dists.append(i * dist_step)

# quality control -----------------------------------------
# remove high values ::
sino[sino >= high_val] = np.nan
sino = pd.DataFrame(sino, dtype=float)
sino = sino.interpolate(
    method='linear',
    axis=1
)

# replace extra nans ::
if src_num == 3:
    sino = sino.fillna(250)

# return to numpy ::
sino = sino.to_numpy()

# plot initial data ---------------------------------------
# get extent ::
start_dist = 0
end_dist = 22
start_angle = 0
end_angle = angles[-1]

# create figure ::
fig, ax = plt.subplots()

# plot ::
ax.imshow(
    sino,
    extent=[start_dist, end_dist, end_angle, start_angle],
    aspect="auto"
)

# set labels ::
ax.set_xlabel("Distance (cm)")
ax.set_ylabel("Angle (degrees)")
ax.set_title("Sinogram")

# show and close ::
plt.show()
plt.close()

# reconstruct ---------------------------------------------
# initialize distance vector ::
expand_vec = np.ones(sino[0].size)

# initialize outer product array ::
expanded_data = []

# produce outer product ::
for item in sino:
    expanded_data.append(np.outer(expand_vec, item))

# # make plot ::
# fig, ax = plt.subplots(nrows=2, ncols=3)

# # plot ::
# ax[0, 0].imshow(expanded_data[0])
# ax[0, 1].imshow(expanded_data[1])
# ax[0, 2].imshow(expanded_data[2])
# ax[1, 0].imshow(expanded_data[3])
# ax[1, 1].imshow(expanded_data[4])
#
# # delete last axis ::
# fig.delaxes(ax[1][2])
#
# # show and close ::
# plt.show()
# plt.close()

# initial rotation ::
rotated_data = []
for index, item in enumerate(expanded_data):
    rotated_data.append(
        ndimage.rotate(
            item,
            angles[index],
            reshape=False,
            order=1)
    )

# # plot initial rotation ::
# fig, (ax, ay) = plt.subplots(2)
# ax.imshow(rotated_data[0])
# ay.imshow(rotated_data[1])
# plt.show()
# plt.close()

# make initial composite and normalized rotation ::
composite = rotated_data[0] * rotated_data[1]
normalized = np.power(composite, (1/2))

# # plot initial normalized composite ::
# fig, ax = plt.subplots()
# ax.imshow(normalized)
# plt.show()
# plt.close()

# make final composite normalization ::
composite = np.ones_like(rotated_data[0])
for item in rotated_data:
    composite *= item
normalized = np.power(composite, (1 / len(rotated_data)))

fig, ax = plt.subplots()
ax.imshow(normalized)
plt.show()
plt.close()

# contour plotting ::
fig, ax = plt.subplots()
x = np.linspace(0, 22, len(normalized))
y = np.linspace(0, 22, len(normalized))
X, Y = np.meshgrid(x, y)
num_lines = 6
flipped = normalized[::-1]
p = ax.contour(X, Y, flipped, num_lines)
ax.clabel(p, inline=True, fontsize=8)
ax.set_aspect('equal')
plt.show()
plt.close()

# full 3d projection ::
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, 22, len(normalized))
y = np.linspace(0, 22, len(normalized))
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, flipped, cmap='plasma')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_zlabel('coincidence intensity')
ax.set_title('3D Projection Example')
plt.show()
plt.close()

# per-slice full 3d projection ::
# composite = np.ones_like(rotated_data[0])
# for index, item in enumerate(rotated_data):
#     fig = plt.figure()
#     ax = fig.add_subplot(211, projection='3d')
#     ay = fig.add_subplot(212, projection='3d')
#     x = np.linspace(0, 22, len(item))
#     y = np.linspace(0, 22, len(item))
#     X, Y = np.meshgrid(x, y)
#     ax.plot_surface(X, Y, item, cmap='plasma')
#     composite *= item
#     ay.plot_surface(X, Y, np.power(composite, (1 / (index + 1))), cmap='plasma')
#     ax.set_xlabel('x (cm)')
#     ax.set_ylabel('y (cm)')
#     ax.set_zlabel('coincidence intensity')
#     ax.set_title('3D Projection Example')
#     ay.set_xlabel('x (cm)')
#     ay.set_ylabel('y (cm)')
#     ay.set_zlabel('coincidence intensity')
#     ay.set_title('3D Reconstruction')
#     plt.show()
#     plt.close()

# Interpolation--------------------------------------------
# duplicate data ::
expansion_value = 2
expanded_slice = [np.array([])]
for index, slices in enumerate(sino):

    # go through each slice ::
    for element in slices:

        # make an x by 1 array full of one value ::
        repeat_values = np.full(expansion_value, element)

        # store value ::
        expanded_slice[index] = np.append(expanded_slice[index], repeat_values)

    # initialize next appended array ::
    expanded_slice.append(np.array([]))
expanded_slice.pop()

# outer product duplicated data ::
expand_vec = np.ones(expanded_slice[0].size)
expanded_rotation = []
for index, item in enumerate(expanded_slice):
    expanded_rotation.append(ndimage.rotate(np.outer(expand_vec,item),angles[index],reshape=False))
expanded_composite = np.ones_like(expanded_rotation[0])
for item in expanded_rotation:
    expanded_composite *= item
expanded_normalized = np.power(expanded_composite,(1/len(data)))
fig, (ax,ay) = plt.subplots(2)
ax.imshow(normalized)
ax.set_title("Multiplication combination: No smoothing")    
ay.imshow(expanded_normalized)
ay.set_title("Multiplication combination: 2x smoothing")
fig.tight_layout()
plt.show()
plt.close()

# Inverse transform ---------------------------------------
# reconstruct ::
reconstruction = iradon(np.transpose(np.array(sino)), theta=np.array(angles))

# plot ::
fig, ax = plt.subplots()
ax.imshow(reconstruction)
ax.set_title("Inverse radon transform data")
plt.show()
plt.close()
