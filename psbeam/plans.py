# """
# OpenCV Bluesky Plans
# """
# ############
# # Standard #
# ############
# import time
# import logging
# import multiprocessing
# ###############
# # Third Party #
# ###############
# import cv2
# import bluesky
# import numpy as np
# from ophyd import Device, Signal
# from bluesky.utils import Msg
# from bluesky.plans import mv, trigger_and_read, run_decorator, stage_decorator

# ##########
# # Module #
# ##########
# import psbeam.preprocessing as prep
# from psbeam.beamexceptions import NoBeamPresent
# import psbeam.beamdetector as psb

# logger = logging.getLogger(__name__)

# # Globals
# circ = cv2.imread("/reg/neh/home5/apra/work/python/psbeam/psbeam/images" \
#                   "/circle_small.png", 0)
# circ_cnt = psb.get_contours(circ, factor=0)[0]

# class Processor(object):
#     def __init__(self,resize=1.0, kernel=(13,13)):
#         self.resize=resize
#         self.kernel=kernel

#     def __call__(self, image):
#         image_prep = prep.uint_resize_gauss(image, kernel=self.kernel)
        
#         try:
#             contours = psb.get_contours(image_prep)

#             contour, area = psb.get_largest_contour(
#                 image_prep, contours=contours, get_area=True)
#             M = psb.get_moments(contour=contour)
#             # Check if beam is in the image using the sum of the pixel values
#             centroid = [pos//self.resize for pos in psb.get_centroid(M)]
#             _, _, l, w = [val//self.resize for val in psb.get_bounding_box(
#                 image_prep, contour)]

#             # import ipdb; ipdb.set_trace()

#             match = cv2.matchShapes(circ_cnt, contour, 1, 0.0)
#             return image.sum(), image.mean(), centroid[0], centroid[1], l, w, match
#         except NoBeamPresent:
#             return None, None, None, None, None, None, None

# def measure_beam_quality(detector, plugin, read_rate=None, num_reads=100):
#     """
#     Measures various stats about the beam.
#     """
#     if read_rate is None:
#         read_rate = min(plugin.array_rate.value/10, 10)
    
#     #Gather shots
#     # logger.debug("Gathering shots..")
#     # #Trigger detector and wait for completion
#     # yield Msg('trigger', detector, group='B')
#     # #Wait for completion
#     # yield Msg('wait', None, 'B')
#     # #Read outputs
#     # yield Msg('create', None, name='primary')
#     det_images = []
#     for i in range(num_reads):
#         time.sleep(1/read_rate)
#         det_images.append(plugin.image)

#     proc = Processor()
#     sums = np.zeros((len(det_images)))
#     means = np.zeros((len(det_images)))
#     centroids_x = np.zeros((len(det_images)))
#     centroids_y = np.zeros((len(det_images)))
#     lengths = np.zeros((len(det_images)))
#     widths = np.zeros((len(det_images)))
#     matches = np.zeros((len(det_images)))

#     for i, image in enumerate(det_images):
#         sums[i], means[i], centroids_x[i], centroids_y[i], lengths[i], widths[i], matches[i] = proc(
#             image)

#     results = {"sum_mn" : sums.mean(),
#                "sum_std" : sums.std(),
#                "mean_mn" : means.mean(),
#                "mean_std" : means.std(),
#                "centroid_x_mn" : centroids_x.mean(),
#                "centroid_x_std" : centroids_x.std(),
#                "centroid_y_mn" : centroids_y.mean(),
#                "centroid_y_std" : centroids_y.std(),
#                "length_mn" : lengths.mean(),
#                "length_std" : lengths.std(),
#                "width_mn" : widths.mean(),
#                "width_std" : widths.std(),
#                "match_mn" : matches.mean(),
#                "match_std" : matches.std()}
#     # yield Msg('save')
#     return results

#     # proc = Processor()
#     # pool = multiprocessing.Pool()
#     # # sums, means, centroids_x, centroids_y, lengths, widths, matches = pool.map(
#     # #     proc, det_images)
#     # results = pool.map(
#     #     proc, det_images)

        
#     # # sums = np.array(sums)
#     # # means = np.array(means)
#     # # centroids_x = np.array(centroids_x)
#     # # centroids_y = np.array(centroids_y)
#     # # lengths = np.array(lengths)
#     # # widths = np.array(widths)
#     # # matches = np.array(matches)

#     # # results = {"sum_mn" : sums.mean(),
#     # #            "sum_std" : sums.std(),
#     # #            "mean_mn" : means.mean(),
#     # #            "mean_std" : means.std(),
#     # #            "centroid_x_mn" : centroids_x.mean(),
#     # #            "centroid_x_std" : centroids_x.std(),
#     # #            "centroid_y_mn" : centroids_y.mean(),
#     # #            "centroid_y_std" : centroids_y.std(),
#     # #            "length_mn" : lengths.mean(),
#     # #            "length_std" : lengths.std(),
#     # #            "width_mn" : widths.mean(),
#     # #            "width_std" : widths.std(),
#     # #            "match_mn" : matches.mean(),
#     # #            "match_std" : matches.std()}
#     # return results


# if __name__ == "__main__":
#     # import pcdsdevices.epics.pim as pim
#     # p = pim.PIM("HFX:DG3:CVV:01")
#     # p = pim.PIM("HX2:SB1:CVV:01")

#     import pcdsdevices.sim.pim as pim
#     p = pim.PIM("TEST")
#     img = cv2.imread("../tests/test_yag_images_01/image97.jpg", 0)
#     p.detector.image1.image = img

#     res = measure_beam_quality(p.detector, p.detector.image1, read_rate=100, num_reads=10)

#     import IPython; IPython.embed()
