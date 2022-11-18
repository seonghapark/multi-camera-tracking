from shapely.geometry.polygon import Polygon, LineString
import numpy as np

class RegionOfInterest():
    """
        Create a polygon representing region of interest
        @coordinates: list of coordinates normalized in [0, 100]
        @image_width: width of actual image
        @image_height: height of actual image
        @road_length: actual road distance (m) of the ROI parallel to the direction of traffic
    """
    def __init__(self, coordinates, image_width, image_height, road_area, road_length, loi_coordinates):
        actual_coordinates = []
        for c in coordinates:
            actual_coordinates.append(
                (int(c[0] * image_width), int(c[1] * image_height))
            )
        self.roi = Polygon(actual_coordinates)
        if self.roi.area < 1:
            raise Exception("Area of the ROI is 0: please check the coordinates {coordinates}}")
        self.width = image_width
        self.height = image_height
        self.road_area = road_area
        self.road_length = road_length
        actual_coordinates = []
        for c in loi_coordinates:
            actual_coordinates.append(
                (int(c[0] * image_width), int(c[1] * image_height))
            )
        self.loi = LineString(actual_coordinates)

    def get_coordinates(self):
        x, y = self.roi.exterior.coords.xy
        a = list(zip(map(int,x.tolist()), map(int,y.tolist())))
        res = [list(ele) for ele in a]
        return np.int32([np.array(res)])

    def get_loi(self):
        x, y = self.loi.xy
        a = list(zip(map(int,x.tolist()), map(int,y.tolist())))
        res = [list(ele) for ele in a]
        return np.int32([np.array(res)])

    def contains_center_of_mass(self, t, b, r, l):
        obj = Polygon([
            (l, t),
            (l, b),
            (r, b),
            (r, t)
        ])
        return self.roi.contains(obj.centroid)

    def loi_intersects(self, t, b, r, l):
        obj = Polygon([
            (l, t),
            (l, b),
            (r, b),
            (r, t)
        ])
        return self.loi.intersects(obj)

    def contains(self, t, b, r, l):
        obj = Polygon([
            (l, t),
            (l, b),
            (r, b),
            (r, t)
        ])
        return self.roi.contains(obj)

    def overlaps(self, t, b, r, l):
        obj = Polygon([
            (l, t),
            (l, b),
            (r, b),
            (r, t)
        ])
        return self.roi.overlaps(obj)