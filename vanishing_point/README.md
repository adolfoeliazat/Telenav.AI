# Vanishing point detector module

#Usage:
#Input is a bgr or grayscale image as a numpy array
#Output is a namedtuple Point(x,y) and a confidence

from vanishing_point.vanishing_point import VanishingPointDetector

vp_detector = VanishingPointDetector()
vp_result, confidence = vp_detector.get_vanishing_point(image)
   
