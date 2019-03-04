#!/usr/bin/env python
# coding: utf-8

# # Getting started
# In this tutorial we will learn the basics of raster and point cloud processing using Pyoints.

# We begin with loading the required modules.

# In[ ]:


import numpy as np

from pyoints import (
	nptools,
	Proj,
	GeoRecords,
	Grid,
	Extent,
	transformation,
	filters,
	clustering,
	classification,
	smoothing,
)


# In[ ]:


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Grid
# We create a two dimensional raster by providing a projection system, a transformation matrix and some data. The transformation matrix defines the origin and scale of the raster. For this example, we just use the default projection system. 

# In[ ]:


print('projection')
proj = Proj()
print(proj)

print('numpy record array:')
rec = nptools.recarray(
	{'indices': np.mgrid[0:50, 0:30].T},
	dim=2
)
print(rec.shape)
print(rec.dtype.descr)

print('transformation matrix')
T = transformation.matrix(t=[-15, 10], s=[0.8, -0.8])
print(T)


# In[ ]:


grid = Grid(proj, rec, T)


# Let's inspect the properties of the raster.

# In[ ]:


print('shape:')
print(grid.shape)
print('number of cells:')
print(grid.count)
print('fields:')
print(grid.dtype)
print('projection:')
print(grid.proj)
print('transformation matrix:')
print(np.round(grid.t, 2))
print('origin:')
print(np.round(grid.t.origin, 2).tolist())
print('extent:')
print(np.round(grid.extent(), 2).tolist())


# Now, we visualize the x 'indices' of the raster.

# In[ ]:


fig = plt.figure(figsize=(10, 10))
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')

plt.imshow(grid.indices[:, :, 0])
plt.show()


# You might have noticed, that the field 'coords' has been implicitly added to the record array. The coordinates correspond to the centers of the raster cells.

# In[ ]:


print(np.round(grid.coords, 2))


# Based on these coordinates we create an additional field representing a surface.

# In[ ]:


x = grid.coords[:, :, 0]
y = grid.coords[:, :, 1]
dist = np.sqrt(x ** 2 + y ** 2)
z = 9 + 10 * (np.sin(0.5 * x) / np.sqrt(dist + 1) + np.cos(0.5 * y) / np.sqrt(dist + 1))
grid = grid.add_fields([('z', float)], data=[z])
print(grid.dtype.descr)


# In[ ]:


fig = plt.figure(figsize=(10, 10))
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')

plt.imshow(grid.z, cmap='coolwarm')
plt.show()


# If we like to treat the raster as a point cloud or list of points, we call the 'records' function. As a result we receive a flattened version of the raster records.

# In[ ]:


print('records type:')
print(type(grid.records()))
print('records shape:')
print(grid.records().shape)
print('coords:')
print(np.round(grid.records().coords, 2))


# We use these flattened coordinates to visualize the centers of the raster cells.

# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(aspect='equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.scatter(*grid.records().coords.T, c=grid.records().z, cmap='coolwarm')
plt.show()


# ## GeoRecords
# The 'Grid' class presented before extends the 'GeoRecords' class. Grid objects, like rasters or voxels are well structured, while GeoRecords in general are just a collection of points.

# To understand the usage of the GeoRecords, we crate a three dimensional point cloud using the coordinates derived before. We also use the same coordinate reference system. The creation of a GeoRecord array requires for a record array with at least a field 'coords' specifying the point coordinates.

# In[ ]:


rec = nptools.recarray({
	'coords': np.vstack([
		grid.records().coords[:, 0],
		grid.records().coords[:, 1],
		grid.records().z
	]).T,
	'z': grid.records().z
})
geoRecords = GeoRecords(grid.proj, rec)


# We inspect the properties of the point cloud first.

# In[ ]:


print('shape:')
print(geoRecords.shape)
print('number of points:')
print(geoRecords.count)
print('fields:')
print(geoRecords.dtype)
print('projection:')
print(geoRecords.proj)
print('transformation matrix:')
print(np.round(geoRecords.t, 2))
print('origin:')
print(np.round(geoRecords.t.origin, 2).tolist())
print('extent:')
print(np.round(geoRecords.extent(), 2).tolist())


# Before we visualize the point cloud, we define the axis limits.

# In[ ]:


axes_lims = Extent([
	geoRecords.extent().center - 0.5 * geoRecords.extent().ranges.max(),
	geoRecords.extent().center + 0.5 * geoRecords.extent().ranges.max()
])


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm')
plt.show()


# ## Transformation
# For some applications we might like to transform the raster coordinates a bit.

# In[ ]:


T = transformation.matrix(t=[15, -10], s=[1.5, 2], r=10*np.pi/180, order='trs')
tcoords = transformation.transform(grid.records().coords, T)


# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(aspect='equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.scatter(*tcoords.T, c=grid.records().z, cmap='coolwarm')
plt.show()


# Or we roto-translate the raster.

# In[ ]:


T = transformation.matrix(t=[1, 2], r=20*np.pi/180)
grid.transform(T)


# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(aspect='equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.scatter(*grid.records().coords.T, c=grid.records().z, cmap='coolwarm')
plt.show()


# ## IndexKD
# The 'GeoRecords' class provides a 'IndexKD' instance to perform spatial neighborhood queries efficiently.

# ### Radial filtering
# We begin with filtering the points within a sphere around some points. As a result, we receive a list of point indices which can be used for sub-sampling.

# In[ ]:


coords = [[-5, 0, 8], [10, -5, 5]]
r = 6.0


# Once in 3D ...

# In[ ]:


fids_list = geoRecords.indexKD().ball(coords, r)
print(len(fids_list))
print(fids_list[0])
print(fids_list[1])


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm', marker='.')
for fids in fids_list:
	ax.scatter(*geoRecords[fids].coords.T, s=100)
plt.show()


# ... and once in 2D.

# In[ ]:


fids_list = geoRecords.indexKD(2).ball(coords, r)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm', marker='.')
for fids in fids_list:
	ax.scatter(*geoRecords[fids].coords.T, s=100)
plt.show()


# Of course, we can do the same with the raster.

# In[ ]:


fids_list = grid.indexKD().ball(coords, r)


# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(aspect='equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.scatter(*grid.records().coords.T, c=grid.records().z, cmap='coolwarm', marker='.')
for fids in fids_list:
	ax.scatter(*grid.records()[fids].coords.T)
plt.show()


# ### Nearest neighbor filtering
# We can also filter the nearest neighbors of the points given before. Next to a list of point indices, we receive a list of point distances of the same shape.

# In[ ]:


k=50


# Once in 3D ...

# In[ ]:


dists_list, fids_list = geoRecords.indexKD().knn(coords, k)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm', marker='.')
for fids in fids_list:
	ax.scatter(*geoRecords[fids].coords.T, s=100)
plt.show()


# ... and once in 2D.

# In[ ]:


dists_list, fids_list = geoRecords.indexKD(2).knn(coords, k)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm', marker='.')
for fids in fids_list:
	ax.scatter(*geoRecords[fids].coords.T, s=100)
plt.show()


# And again, once with the raster.

# In[ ]:


dists_list, fids_list = grid.indexKD(2).knn(coords, k)


# In[ ]:


fig = plt.figure(figsize=(10, 10))
ax = plt.axes(aspect='equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

ax.scatter(*grid.records().coords.T, c=geoRecords.z, cmap='coolwarm', marker='.')
for fids in fids_list:
	ax.scatter(*grid.records()[fids].coords.T)
plt.show()


# ### Point counting
# We have the option to count the number of points within a given radius. For this purpose we select a subset of the raster first. Then, using the point cloud, we count the number of raster cells within the given radius.

# In[ ]:


grid_subset = grid[15:25, 30:40]
count = grid_subset.indexKD(2).ball_count(r, geoRecords.coords)
print(count)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=count, cmap='YlOrRd')
ax.scatter(*grid_subset.records().coords.T, color='black')
ax.scatter(*grid.records().coords.T, color='gray', marker='.')
plt.show()


# You might have noticed, that this was a fist step of data fusion, since we related the raster cells to the point cloud. This can also be done for nearest neighbor or similar spatial queries regardless of dimension.

# ## Point filters
# To create a subset of points, we typically use some kind of point filters. We begin with a duplicate point filter. To ease the integration of such filters into your own algorithms, an iterator is returned instead of a list of point indices.

# In[ ]:


fids = list(filters.ball(geoRecords.indexKD(), 2.5))
print(fids)
print(len(fids))


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm')
ax.scatter(*geoRecords[fids].coords.T, color='red', s=100)
plt.show()


# Sometimes we like to filter local maxima of an attribute using a given radius ...

# In[ ]:


fids = list(filters.extrema(geoRecords.indexKD(2), geoRecords.z, 1.5))
print(fids)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm')
ax.scatter(*geoRecords[fids].coords.T, color='red', s=100)
plt.show()


# ... or find local minima.

# In[ ]:


fids = list(filters.extrema(geoRecords.indexKD(2), geoRecords.z, 1.5, inverse=True))
print(fids)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm')
ax.scatter(*geoRecords[fids].coords.T, color='blue', s=100)
plt.show()


# ## Smoothing
# To compensate for noise, or receive just a smoother result we can use smoothing algorithms. The algorithm presented here averages the coordinates of the nearest neighbors.

# In[ ]:


scoords = smoothing.mean_knn(geoRecords.coords, 20, num_iter=3)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

ax.scatter(*geoRecords.coords.T, c=geoRecords.z, cmap='coolwarm', marker='.')
ax.plot_trisurf(*scoords.T, cmap='gist_earth')
plt.show()


# ## Clustering
# A common problem is to cluster point clouds. Here we use a clustering algorithm, which assigns points iteratively to the most dominant class within a given radius. By iterating from top to bottom, the points are assigned to the hills of the surface.

# In[ ]:


order = np.argsort(geoRecords.z)[::-1]
cluster_indices = clustering.majority_clusters(geoRecords.indexKD(), 5.0, order=order)
print(cluster_indices)
cluster_dict = classification.classes_to_dict(cluster_indices)
print(cluster_dict)


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlim(axes_lims[0], axes_lims[3])
ax.set_ylim(axes_lims[1], axes_lims[4])
ax.set_zlim(axes_lims[2], axes_lims[5])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

for fids in cluster_dict.values():
	ax.scatter(*geoRecords[fids].coords.T, s=100)
plt.show()

