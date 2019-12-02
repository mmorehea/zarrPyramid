import sys
import glob
import numpy as np
import zarr
import code
from itertools import combinations
from functools import reduce
import numpy as np
import os

# credit to https://towardsdatascience.com/countless-3d-vectorized-2x-downsampling-of-labeled-volume-images-using-python-and-numpy-59d686c2f75
# fast pure python downsampling of numpy array, happy to switch out for some other library
def countless(data):
  sections = []
  # shift zeros up one so they don't interfere with bitwise operators
  # we'll shift down at the end
  data += 1 
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)
  pick = lambda a,b: a * (a == b)
  lor = lambda x,y: x + (x == 0) * y # logical or
  subproblems2 = {}
  results2 = None
  for x,y in combinations(range(7), 2):
    res = pick(sections[x], sections[y])
    subproblems2[(x,y)] = res
    if results2 is not None:
      results2 = lor(results2, res)
    else:
      results2 = res
  subproblems3 = {}
  results3 = None
  for x,y,z in combinations(range(8), 3):
    res = pick(subproblems2[(x,y)], sections[z])
    # 7 is the terminal element, we don't need to memoize these solutions
    if z != 7: 
      subproblems3[(x,y,z)] = res
    if results3 is not None:
      results3 = lor(results3, res)
    else:
      results3 = res
  results4 = ( pick(subproblems3[(x,y,z)], sections[w]) for x,y,z,w in combinations(range(8), 4) )
  results4 = reduce(lor, results4) 
  final_result = reduce(lor, (results4, results3, results2, sections[-1])) - 1
  data -= 1
  return final_result

def test(directory_path):
	base = zarr.open(directory_path, mode='r+')
	downsize_dimensions = np.asarray(base.shape) / np.asarray(base.chunks)
	if (np.unique(downsize_dimensions).size != 1):
		print("not all dimensions reduce equally; should never happen?")
	downsize_factor = int(downsize_dimensions[0]**(1/2))
	
	downsize_factor -= 1 # already have first level of pyramid
	small_image = base[:] #currently must fit into RAM, needs to scale
	levels = []

	while(downsize_factor >= 0):
		small_image = countless(small_image)
		newLevel = zarr.array(small_image, chunks=base.chunks)
		zarr.save(os.path.join(directory_path, str(downsize_factor)), newLevel)
		downsize_factor -= 1


def main():
	print("zarrPyramid, by Michael Morehead")
	print("pyramid experiment")

	print("---------------------------------------")
	print("Usage: python zarrPyramid.py [path/to/zarr/directory]")
	print("---------------------------------------")
	path = sys.argv[1]

	test(path)

if __name__== "__main__":
	main()