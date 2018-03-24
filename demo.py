# -*- coding: utf-8 -*-
"""
============================
Usage demo for fun4elli.py
============================

Demonstrates the usage of some of the functions in
`fun4elli.py`.

"""

import numpy as np
from matplotlib import pyplot as plt
import fun4elli

# Makes points to be fitted to an ellipse
tParam = (10., 20., 10., 7., np.pi/6);
n = 60;
pts = fun4elli.getPoints(tParam,
                np.linspace(0, 2*np.pi, n, endpoint=False))
print("Number of data points", n)

# Tests fitting function
tConicHat = fun4elli.fitEllipseDirect(pts)
tParamHat = fun4elli.conic2Param(tConicHat)
print("Ellipse estimation errors:", np.subtract(tParamHat, tParam))
assert(np.allclose(tParam, tParamHat))

# Round point coordinates
pts = np.round(pts)

# Calculates distances from points to ellipse
dists, projes = fun4elli.distEllipsePoints(tParam, pts, 
                                           signed=True, getProjes=True)

# Checks if points with negative distances are inside the ellipse
rpe = fun4elli.relationPointsEllipse(pts, fun4elli.param2FociStr(tParam))
print("Sign matches :", np.all(np.sign(dists) == rpe) )
assert(np.all(np.sign(dists) == rpe))

# Fit Rounded points
tConicHat = fun4elli.fitEllipseDirect(pts)
tParamHat = fun4elli.conic2Param(tConicHat)
print("Ellipse estimation errors:", np.subtract(tParamHat, tParam))

# Gets the points to draw the ellipses
ellipse = fun4elli.getPoints(tParam, np.linspace(0, 2*np.pi, 60))
ellipseH = fun4elli.getPoints(tParamHat, np.linspace(0, 2*np.pi, 60))

# Visualizes results
plt.figure()
plt.plot(pts[:, 0], pts[:, 1], '.g')
plt.plot(ellipse[:, 0], ellipse[:, 1], 'g')
plt.plot(ellipseH[:, 0], ellipseH[:, 1], 'b')
for (x0, x1), (y0, y1) in zip(pts, projes):
    plt.plot([x0, y0], [x1, y1], 'g')
plt.axis('equal')
plt.show()
