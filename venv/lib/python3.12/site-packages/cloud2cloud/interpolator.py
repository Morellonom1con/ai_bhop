"""
Interpolator
============

cloud2cloud implementation and API.
"""
import numpy as np
from scipy import spatial


SMALL = 1e-16
class CloudInterpolator:
	"""Wrapper around a source and target meshe to facilitate interpolation between the two."""
	def __init__(self, source, target, limitsource=None, stencil=4, function=None):
		"""
		Initialisation

		:param source: ndarray(dim_msh, points_sce) or tuple(ndarray(points_sce),...)
		:param target: ndarray(dim_msh, points_tgt) or tuple(ndarray(points_tgt),...)
		:param limitsource: the maximum number of points to use for the interpolation
		:param stencil: the number of neighbours to use for the interpolation
		:param function: determine the coefficient to give to each neighbours from their distance (default is linear)
		"""
		n_dim = len(source)
		if n_dim != len(target):
			print("Warning: source and target dim mismatch")
			pass #error "mismatch dims"

		n_points = source[0].size
		self.skipper = 1
		if limitsource is not None and n_points > limitsource:
			self.skipper = 1+(n_points-1)//limitsource

		if isinstance(source, np.ndarray):
			source = np.stack(source[:,::self.skipper], axis=1)
		else:
			source = np.stack([axe[::self.skipper] for axe in source], axis=1)
		target = np.stack(target, axis=1)
		kdtree = spatial.cKDTree(source)
		dists, self.index = kdtree.query(target, k=stencil)

		if stencil == 1:
			self.index = self.index[:,None]
			self.weight = np.ones((self.index.size, 1))
			return

		if function is not None:
			dists[...] = function(dists)
		dists[...] = np.reciprocal(np.maximum(dists, SMALL))
		dists /= np.sum(dists, axis=1)[:,None]
		self.weight = dists

	def interp(self, data):
		"""
		Interpolate data bewteen the source and target meshes.

		:param data: ndarray(points_sce, \*shape_val)
		:returns: ndarray(points_tgt, \*shape_val)
		"""
		estimate = data[::self.skipper,...][self.index]
		estimate *= self.weight.reshape(*self.weight.shape, *[1]*(data.ndim-1))
		return np.sum(estimate, axis=1)


def cloud2cloud(source_msh, source_val, target_msh, unroll_axis=None, verbose=False, **kwargs):
	"""
	Interpolate source_val between source_msh and target_msh.

	:param source: ndarray(\*shape_sce, dim_msh)
	:param target: ndarray(\*shape_tgt, dim_msh)
	:param values: ndarray(\*shape_sce, \*shape_val)
	:param verbose: if True print shape information
	:param kwargs: key word arguments forwarded to CloudInterpolator
	:returns: ndarray(\*shape_tgt, \*shape_val)
	"""
	*shp_sce, dim_sce = source_msh.shape
	*shp_tgt, dim_tgt = target_msh.shape
	shp_sce, shp_tgt = tuple(shp_sce), tuple(shp_tgt)
	shp_val = source_val.shape[len(shp_sce):]
	n_p_sce = np.prod(shp_sce)
	n_p_tgt = np.prod(shp_tgt)

	if verbose:
		if dim_sce != dim_tgt:
			print("Warning: source and target dim mismatch")
		if source_val.shape[:len(shp_sce)] != shp_sce:
			print("Warning: Source and data mismatch")
		print("dim_sce:", dim_sce)
		print("dim_tgt:", dim_sce)
		print("shp_sce:", shp_sce)
		print("shp_tgt:", shp_tgt)
		print("shp_val:", shp_val)
		print("n_p_sce:", n_p_sce)
		print("n_p_tgt:", n_p_tgt)

	source_val = source_val.reshape(n_p_sce, *shp_val)
	source_msh = source_msh.reshape(n_p_sce, dim_sce)
	target_msh = target_msh.reshape(n_p_tgt, dim_tgt)

	if verbose:
		print("new shp_sce:", source_msh.shape)
		print("new shp_tgt:", target_msh.shape)
		print("new shp_val:", source_val.shape)

	base = CloudInterpolator(source_msh.T, target_msh.T, **kwargs)
	if verbose and base.skipper > 1:
		print("skipper:", base.skipper)

	if unroll_axis is None:
		return base.interp(source_val).reshape(*shp_tgt, *shp_val)
	else:
		axis_size = shp_val[unroll_axis]
		shp_partial_val = tuple(_ for i,_ in enumerate(shp_val) if i!=unroll_axis)
		shp_partial_tgt = (slice(None),)*len(shp_tgt)
		shp_partial_pts = (slice(None),)
		result = np.empty((*shp_tgt, *shp_val))
		if verbose:
			print(axis_size, shp_val, shp_partial_val)
		for i in range(axis_size):
			partial_indexes = tuple(i if j==unroll_axis else slice(None) for j in range(len(shp_val)))
			if verbose:
				print(i, partial_indexes, shp_partial_tgt+partial_indexes, shp_partial_pts+partial_indexes)
			result[shp_partial_tgt+partial_indexes] = base.interp(source_val[shp_partial_pts+partial_indexes]).reshape(*shp_tgt, *shp_partial_val)
		return result
