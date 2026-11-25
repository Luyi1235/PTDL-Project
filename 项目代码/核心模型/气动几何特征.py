"""Geometry-only interaction features derived from the actual airframe construction."""

import numpy as np


def interference_features(x):
    x = np.atleast_2d(np.asarray(x, float))
    area, ar, taper, sweep, twist, length, diameter = x.T
    span = np.sqrt(area*ar)
    root = 2*area/(span*(1+taper))
    mac = 2/3*root*(1+taper+taper*taper)/(1+taper)
    tail_area = .65*area*mac/(.52*length)
    tail_span = np.sqrt(4*tail_area)
    tail_root = 2*tail_area/(1.5*tail_span)
    separation = .48*length
    height = .2*diameter
    # Distances and areas follow the wing/tail placement and tail-volume equations.
    derived = np.column_stack((
        ar, taper, sweep, twist, length/span, diameter/mac, diameter/span, mac/length,
        diameter/length, root/span, tail_area/area, tail_span/span, tail_root/mac,
        separation/mac, height/mac, np.rad2deg(np.arctan2(height, separation)),
        (separation-.25*mac)/tail_root, (separation+.25*tail_root)/mac,
        np.tan(np.deg2rad(sweep))*span/(2*separation),
        np.tan(np.deg2rad(3.))*tail_span/(2*height),
        twist*(1+2*taper)/(3*(1+taper)),
    ))
    return np.column_stack((x, derived))
