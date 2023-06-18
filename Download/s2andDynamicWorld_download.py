# coding:utf-8
import argparse
import csv
import math
from multiprocessing.dummy import Pool, Lock
import os
import datetime as DATETIME
import random
import geemap
# datetime.strptime(str(a),"%Y%m%d")
import warnings

warnings.simplefilter('ignore', UserWarning)
import ee
import numpy as np
import urllib3

ALL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
RGB_BANDS = ['B4', 'B3', 'B2']
RGB_BANDS_NIR = ['B4', 'B3', 'B2', 'B8']
dates = [2016,
         2017,
         2018,
         2019,
         2020,
         2021]


class UniformSampler:

    def sample_point(self):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-90, 90)
        return [lon, lat]


def read_csv(file):
    points = []
    with open(file, encoding="unicode_escape") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lon = row[4]
            lat = row[5]
            if lon == "x":
                continue
            points.append([float(lon), float(lat)])
    return points


class GaussianSampler:

    def __init__(self, interest_points=None, std=150, size=None):
        self.interest_points = interest_points
        self.std = std
        self.size = size
        self.count = 0

    def initialize_point(self):
        center_point = self.interest_points[self.count]
        self.count += 1
        return center_point

    def sample_one_point(self, center_point):
        std = self.km2deg(self.std)
        lon, lat = np.random.normal(loc=center_point, scale=[std, std])  # Gaussian Sample
        # lon, lat = self.getRandomPointInCircle(radius=std, centerx=center_point[0],
        #                                        centery=center_point[1])  # Circle sample
        return [lon, lat]

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)

    @staticmethod
    def getRandomPointInCircle(radius, centerx, centery):
        theta = random.random() * 2 * np.pi
        r = random.uniform(0, radius ** 2)
        x = math.cos(theta) * (r ** 0.5) + centerx
        y = math.sin(theta) * (r ** 0.5) + centery

        return x, y


def downloadS2andDynamicWorld(sampler, dates, center_coord, sub_location_path, cloud_pct, debug=False):
    coord = sampler.sample_one_point(center_coord)
    periods = get_period(dates)
    # We use the size of 64*64 samples for training, so the radius of the buffer is 320 meters (10 meters resolution)
    loc = ee.Geometry.Point(coord).buffer(320).bounds()
    try:
        for period in periods:
            s2_collection = (
                ee.ImageCollection('COPERNICUS/S2')
                    .filterBounds(loc)
                    .filterDate(period[0], period[1])
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                    .map(maskS2clouds)
            )
            dynamicWorld_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1"). \
                filterDate(period[0], period[1]). \
                filterBounds(loc)
            s2_image = s2_collection.median().select(['B2', 'B3', 'B4', "B8"]).clip(loc).divide(10000.0)
            dynamicWorld_image = dynamicWorld_collection.median().select(["built"]).clip(loc)
            geemap.download_ee_image(s2_image, f"{sub_location_path}/{period[0].split('-')[0]}_s2.tif", scale=10,
                                     region=loc,
                                     crs="EPSG:4326")  # EPSG:4326 is WGS84
            geemap.download_ee_image(dynamicWorld_image,
                                     f"{sub_location_path}/{period[0].split('-')[0]}_dynamicWorld.tif", scale=10,
                                     region=loc,
                                     crs="EPSG:4326")
    # Sometimes the network is disconnected (more noticeable in mainland China), so it needs to be re-run
    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        downloadS2andDynamicWorld(sampler, dates, center_coord, sub_location_path, cloud_pct, debug=debug)


def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    return image.updateMask(mask)

def get_period(dates):
    year1, year2, year3 = random.sample(dates, 3)
    time1_0 = DATETIME.date(year1, 1, 1)
    time1_1 = DATETIME.date(year1, 12, 30)
    time2_0 = DATETIME.date(year2, 1, 1)
    time2_1 = DATETIME.date(year2, 12, 30)
    time3_0 = DATETIME.date(year3, 1, 1)
    time3_1 = DATETIME.date(year3, 12, 30)
    return [(time1_0.isoformat(), time1_1.isoformat()), (time2_0.isoformat(), time2_1.isoformat()),
            (time3_0.isoformat(), time3_1.isoformat())]


if __name__ == '__main__':
    # For users in mainland China, they need to use a VPN and configure a proxy here, because google services are not
    # accessible in China
    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=r"D:\try", help="Save path")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for multiprocessing")
    parser.add_argument('--sample_points', type=int, default=200,
                        help="Number of sampling points around the base point ")
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    ee.Authenticate()
    ee.Initialize()
    # A file containing coordinates of more than 2,
    # 000 administrative units in China to provide basic sampling information. If you want to sample in other
    # countries, you can prepare a similar document
    points = read_csv(r"./county_locations.csv")
    n = args.sample_points
    # Use Gaussian sampling to keep data concentrated in urban areas as much as possible
    sampler = GaussianSampler(interest_points=points)


    def worker(idx):
        idx = idx
        try:
            center_coord = sampler.initialize_point()
        except Exception as e:  # Sampling overruns
            print(e)
            return
        for loc_id in range(n):
            if args.save_path is not None:
                location_path = os.path.join(args.save_path, f'{idx:06d}')
                os.makedirs(location_path, exist_ok=True)
                sub_location_path = os.path.join(location_path, str(loc_id))
                os.makedirs(sub_location_path, exist_ok=True)
                downloadS2andDynamicWorld(sampler, dates, center_coord, sub_location_path, args.cloud_pct, args.debug)
        return


    indices = range(len(points))

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
