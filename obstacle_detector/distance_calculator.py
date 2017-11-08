import csv

from scipy.interpolate import interp1d
from functools import lru_cache

class Distance_calculator:
	def __init__(self, pxs, meters):
		self.__interpolate = interp1d(
			pxs, meters, kind='linear')
		self.__deinterpolate = interp1d(
			meters, pxs, kind='linear')

	@lru_cache(maxsize=360)
	def get_rails_px_height_by_distance(self, y_coord):
		return self.__deinterpolate(y_coord)

	@lru_cache(maxsize=360)
	def get_distance_by_rails_px_height(self, y_coord):
		return self.__interpolate(y_coord)


with open('data/spline-data.csv') as csv_file:
	spline_data = csv.reader(csv_file)
	pxs, meters = zip(*spline_data)
	pxs = list(map(lambda x: int(x), pxs))
	meters = list(map(lambda x: float(x), meters))

spline_dist = Distance_calculator(pxs, meters)

