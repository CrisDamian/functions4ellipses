# -*- coding: utf-8 -*-
"""
===========
fun4elli.py
===========
Copyright (c) 2016 Cristian Damian

Created on Thu Jul 21 16:46:40 2016

This file contains different functions related to ellipses. This
includes conversions of ellipse parameters, fitting and finding distances.

Dependencies
============
Numpy and Scipy packages.
If run as a script uses the Matplotplib package too.

Representations of ellipses
===========================

In order to represent ellipses we use tuples that carry the paramaters of a
particular representation method. When refering to such a tuple I will use the
phrase " x representation touple" as a shorthand.

There are 3 ways to represent an ellipse:

tConic : (A, B, C, D, E F)
        The conic represientation touple of the ellipse. It consists of the
        coeffficients for the 2D quadratic equation that describes the ellipse.
        The equation has the following form::

             A*x**2+B*x*y+C*y**2+D*x+E*y+F==0

tParam : (cx, cy, l, w, t)
        The parametric representation touple of the ellipse. It consists of
        the coefficients for the parametric representation of the ellipse.
        The parameters have the following meanings:
        `cx, cy` are the coordinates ot the center of the ellipse,
        `l` is the length of the major axis (the length) of the ellipse,
        `w` is the length of the minor axis (the width) of the ellipse,
        `t` is the tilt of the ellipse in radians.

tFociStr : (x1, y1, x2, y2, s)
        The foci and string representation touple of the ellipse.
        It constists of:
        `x1, y1, x2, y2` the cartesian coordinates of the 2 foci of the ellipse,
        `s` the length of the string used to draw the ellipse.


References
----------
Van Loan, C.F., 2008. Using the Ellipse to Fit and Enclose Data Points.
Department of Computer Science Cornell University, p.54.

A. Fitzgibbon, M. Pilu and R. B. Fisher, 1999. Direct least square fitting
of ellipses. IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 21, no. 5, pp. 476-480. DOI: 10.1109/34.765658

D. Eberly, 2013. Distance from a point to an ellipse, an ellipsoid, or a
hyperellipsoid. Geometric Tools, LLC. URL:
http://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf

---------------

The MIT License (MIT)

Copyright (c) 2016 Cristian Damian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
from scipy import linalg
from scipy import optimize


def conic2Param(tConic):
    """
    Recieves the conic representation tuple of an ellipse and returns a
    parametric representation touple.

    Parameters
    ----------
    tConic : (A, B, C, D, E F)
        The conic represientation touple of the ellipse. It consists of the
        coeffficients for the 2D quadratic equation that describes the ellipse.
        The equation has the following form::

             A*x**2+B*x*y+C*y**2+D*x+E*y+F==0

    Returns
    -------
    tParam : (cx, cy, l, w, t)
        The parametric representation touple of the ellipse. It consists of
        the coefficients for the parametric representation of the ellipse.
        The parameters have the following meanings:
        `cx, cy` are the coordinates of the center of the ellipse,
        `l` is the length of the major axis of the ellipse,
        `w` is the length of the minor axis of the ellipse,
        `t` is the tilt of the ellipse in radians.

    Raises
    ------
    LinAlgError
        Eigenvalue computation does not converge. The conversion requires an
        eigenvalue computation done with the scipy library.

    """

    # Auxiliary variables
    A, B, C, D, E, F = tConic

    M0 = np.matrix([[F, D/2, E/2],
                    [D/2, A, B/2],
                    [E/2, B/2, C]])
    M = np.matrix([[A, B/2],
                   [ B/2, C]]);

    detM = np.linalg.det(M)
    detM0 = np.linalg.det(M0)

    # Eigenvalue problem
    eigval, eigvec = linalg.eig(M)
    if np.abs(eigval[0]-A)>np.abs(eigval[0]-C):
        eigval = np.flipud(eigval)
        eigvec = np.fliplr(eigvec)

    # Parameters
    cx = (B*E-2*C*D)/(4*A*C-B*B)
    cy = (B*D-2*A*E)/(4*A*C-B*B)
    l = np.sqrt(abs(np.real(-detM0 / (detM*eigval[0]))))
    w = np.sqrt(abs(np.real(-detM0 / (detM*eigval[1]))))
    tau = np.arctan(B/(A-C))/2

    return (cx, cy, l, w, tau)


def param2Conic(tParam):
    """
    Recieves the parametric representation tuple of an ellipse and returns
    the conic representation touple.

    Parameters
    ----------
    tParam : (cx, cy, l, w, t)
        The parametric representation touple of the ellipse. It consists of
        the coefficients for the parametric representation of the ellipse.
        The parameters have the following meanings:
        `cx, cy` are the coordinates ot the center of the ellipse,
        `l` is the length of the major axis of the ellipse,
        `w` is the length of the minor axis of the ellipse,
        `t` is the tilt of the ellipse in radians.

    Returns
    -------
    tConic : (A, B, C, D, E F)
        The conic represientation touple of the ellipse. It consists of the
        coeffficients for the 2D quadratic equation that describes the ellipse.
        The equation has the following form::

             A*x**2+B*x*y+C*y**2+D*x+E*y+F==0
    """
    #Auxiliary vaiables
    cx, cy, l, w, tau = tParam
    s=np.sin(tau)
    c=np.cos(tau)

    # Conic parameters
    A = (w*c)**2+(l*s)**2
    B = 2*c*s*(l**2-w**2)
    C = (w*s)**2+(l*c)**2
    D = -2*A*cx - cy*B
    E = -2*C*cy - cx**B
    F = -(l*w)**2 + A*cx**2 + B*cx*cy + C*cy**2

    return (A, B, C, D, E, F)


def param2FociStr(tParam):
    """
    Recieves the parametric representation tuple of an ellipse and returns
    the foci and string representation touple.

    Parameters
    ----------
    tParam : (cx, cy, l, w, t)
        The parametric representation touple of the ellipse. It consists of
        the coefficients for the parametric representation of the ellipse.
        The parameters have the following meanings:
        `cx, cy` are the coordinates ot the center of the ellipse,
        `l` is the length of the major axis of the ellipse,
        `w` is the length of the minor axis of the ellipse,
        `t` is the tilt of the ellipse in radians.

    Returns
    -------
    tFociStr : (x1, y1, x2, y2, s)
        The foci and string representation touple of the ellipse.
        It constists of:
        `x1, y1, x2, y2` the cartesian coordinates of the 2 foci of the ellipse,
        `s` the length of the string used to draw the ellipse.

    """
    #Auxiliary variables
    cx, cy, l, w, tau = tParam
    sinT=np.sin(tau)
    cosT=np.cos(tau)
    c = (l**2-w**2)**.5

    # Foci&String parameters
    x1 = cx - cosT*c
    y1 = cy - sinT*c
    x2 = cx + cosT*c
    y2 = cy + sinT*c
    s = 2*l

    return (x1, y1, x2, y2, s)


def fociStr2Param(tFociStr):
    """
    Recieves the foci and string representation touple of an ellipse and
    returns the parametric representation tuple.

    Parameters
    ----------
    tFociStr : (x1, y1, x2, y2, s)
        The foci and string representation touple of the ellipse.
        It constists of:
        `x1, y1, x2, y2` the cartesian coordinates of the 2 foci of the ellipse,
        `s` the length of the string used to draw the ellipse.

    Returns
    -------
    tParam : (cx, cy, l, w, t)
        The parametric representation touple of the ellipse. It consists of
        the coefficients for the parametric representation of the ellipse.
        The parameters have the following meanings:
        `cx, cy` are the coordinates ot the center of the ellipse,
        `l` is the length of the major axis of the ellipse,
        `w` is the length of the minor axis of the ellipse,
        `t` is the tilt of the ellipse in radians.

    """
    # Auxiliary variables
    x1, y1, x2, y2, s = tFociStr

    # Parameters
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    l = s/2
    w = (s**2 -((x1-y1)**2+(y1-y2)**2))**.5/2
    t = np.arctan2((y2-y1),(x2-x1))

    return (cx, cy, l, w, t)


def getPoints(tParam, theta):
    """
    Returns the Cartesian coordinates of the points on the ellipse expressed
    by a parametric representation touple that have the given parametric
    angles.

    Parameters
    ----------
    tParam : (cx, cy, l, w, t)
        Parametric represenation tuple of an ellipse.

    theta : numpy.array_like
        Parametric angles of the points.

    Returns
    -------

    res : numpy.ndarray
        A ``n*2``  array with the Cartesian coordinates of the points.
        Column 0 is has the X coordinates and column 1 has the Y coordinates.

    """
    cx, cy, l, w, t = tParam
    cost = np.cos(t)
    sint =np.sin(t)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    x = cx + cost * l * costheta - sint * w * sintheta
    y = cy + sint * l * costheta + cost * w * sintheta
    return np.vstack((x, y)).T



def fitEllipseDirect(points):
    """
    Fits an ellipse to the given points using the method from
    Fitzgibbon et al. (1991).

    Parameters
    ----------
    points : numpy.array_like
        A n*2  array with the coordinates of the points to which the ellipse
        will be fitted. Column 0 has the X coordinates and column 1 has the Y
        coordinates.

    Returns
    -------
    tConic : (A, B, C, D, E, F)
        The conic representation touple of the fitted ellipse.

    Raises
    ------
    LinAlgError
        Eigenvalue computation does not converge.

    References
    ----------
    A. Fitzgibbon, M. Pilu and R. B. Fisher. 1999. Direct least square fitting
    of ellipses. IEEE Transactions on Pattern Analysis and Machine
    Intelligence, vol. 21, no. 5, pp. 476-480. DOI: 10.1109/34.765658

    """
    x = points[:, 0]; y = points[:, 1];
    # Build design matrix
    D = np.vstack((x*x, x*y, y*y, x, y, np.ones(x.shape)))
    # Build scatter matrix
    S = D.dot(D.T)
    # Build constraint matrix
    C = np.zeros((6, 6))
    C[0, 2]= +2; C[1, 1]= -1; C[2, 0]= +2;
    # Solve generalised eigenvalue system C*a == l*S*a
    geval, gevec = linalg.eig(S, C)
    # Find the eigenvector with the only pozitive eigenvalue
    geval = np.real(geval)
    i = np.argmax((geval>0) * np.isfinite(geval))
    if not np.isfinite(geval[i]):
        raise linalg.LinAlgError(
                "Eigenvalue calculation failed to return a valid answer." +
                "\nEigenvalues:\n" + str(geval) + '\n')
    theVec = np.real(gevec[:, i])
    # That vector has the parameters of the ellipse
    return tuple(theVec.flatten())


def _projectPointEllipse_(a, b, y0, y1, tol):
    """
    Private function. 
	This is the heart of the distance finding algorithm from `distEllipsePoints()`.
    It is a translation of Listing 1 from Eberly (2013).
    """
    if y0>tol and y1>tol:
        fObj = lambda t: (a*y0/(t+a**2))**2 +  (b*y1/(t+b**2))**2 - 1
        tmin = -b**2 + b*y1
        tmax = -b**2 + np.sqrt(a**2*y0**2+b**2*y1**2)
        tbar = optimize.brentq(fObj, tmin, tmax);
        x0 = a**2*y0/(tbar+a**2);
        x1 = b**2*y1/(tbar+b**2);
        d = np.sign(tbar) * np.sqrt((x0-y0)**2 + (x1-y1)**2)

    elif y1 > tol:  #and y0==0
        x0 = 0
        x1 = b
        d = (y1-x1)


    elif y0 > (a**2-b**2)/a:  #and  y1==0
        x0 = a
        x1 = 0
        d = (y0-x0)

    else: #if y1==0 and y0<thresh
        x0 = a**2*y0/(a**2-b**2)
        x1 = b*np.sqrt(1- (x0/a)**2)
        d = np.sqrt( (x0-y0)**2 + x1**2)

    return d, (x0, x1)


def distEllipsePoints(tParam, points, signed=False, getProjes  = False, tol=None):
    """
    Finds the distances from an ellipse to a set of points using the method
    of Eberly (2013).

    Parameters
    ----------
    tParam : (cx, cy, l, w, t)
        Parametric represenation tuple of the ellipse.

    points : numpy.array_like
        A n*2 array with the coordinates of the points to which to compute the
        distance. Column 0 has the X coordinates and column 1 has the Y
        coordinates.

    signed : boolean, optional
        Default is `False`. If `True` the sign of the distance will be
        negative for the points that are inside the ellipse.

    getProjes : boolean, optional
        Default is `False`. If `True` the function retuns  the tuple
        `(dists, projes)`.

    Returns
    -------
    dists : numpy.ndarray
        A 1d array with the distances from the ellipse to the points.

    projes : numpy.ndarray
        Returns the closest points to the imput points that lie on the ellipse.
        The format is the same as for `points`.

    References
    ----------

    D. Eberly, 2013. Distance from a point to an ellipse, an ellipsoid, or a
    hyperellipsoid. Geometric Tools, LLC. URL:
    http://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf

    """
    # Handling arguments
    cx, cy, l, w, tau = tParam
    c = (cx, cy);
    points=np.array(points, dtype='float')

    if not (l > 0 and w > 0):
        raise ValueError("Elements 2 and 3 of `tParam` must be pozitive.")

    if not (points.ndim == 2 and points.shape[1] == 2):
        raise ValueError(
            "`points` must be a 2d array and it must have 2 columns.")
    if tol is None:
        tol = np.finfo(float).eps**.5
    # Center axes on ellipse
    points -= c

    # Handling circle case
    isCircle = (abs(l-w) < l*tol)
    if isCircle:
        projes = points*l/np.sum((points)**2, axis=1, keepdims=True)**.5
        dists = np.sum((points)**2, axis=1)**.5 - l

    # Handling ellipse case
    else:
        #Make sure l > w
        if l<w:
            l, w = w, l
            tau += np.pi/2;

        #Move points to ellipse space
        rotMat= np.array([[np.cos(-tau), -np.sin(-tau)],
                           [np.sin(-tau), np.cos(-tau)]])
        points = np.dot(points, rotMat.T)

        #Reflect to first quadrant
        if getProjes: sgnPoints =  np.sign(points)
        points = abs(points);

        #Find distance for every point
        dists = np.empty(points.shape[0])
        projes =  np.empty(points.shape)
        for index in range(dists.size):
            y0, y1 = points[index]
            dists[index], projes[index] = _projectPointEllipse_(l, w, y0, y1, tol)

    if not signed: dists = abs(dists)

    if getProjes:
        # Inverse coordinate change for projes
        if isCircle:
            projes += c
        else:
            projes = np.dot(projes*sgnPoints, rotMat) + c
        return dists, projes
    else:
        return dists


def relationPointsEllipse(points, tFociStr):
    """
    Returns an array with
    -1 for every point inside the ellipse,
    0 for every point on the ellipse and
    +1 for every point outside the ellipse.

    Parameters
    ----------
    points : numpy.array_like
        A n*2 array with the coordinates of the points to which to compute the
        distance. Column 0 has the X coordinates and column 1 has the Y
        coordinates.

    tFociStr : (x1, y1, x2, y2, s)
        The foci and string representation touple of the ellipse.
        It constists of:
        `x1, y1, x2, y2` the cartesian coordinates of the 2 foci of the ellipse,
        `s` the length of the string used to draw the ellipse.

    Returns
    -------
    res : numpy.ndarray
        A 1d array with -1, 0 or +1 for the points that are inside,
                on or outside the ellipse respectively.

    """
    x1, y1, x2, y2, s = tFociStr
    f1 = np.array([x1, y1])
    f2 = np.array([x2, y2])
    resid = ( np.sum((points-f1)**2, axis=1)**.5 +
              np.sum((points-f2)**2, axis=1)**.5 - s )
    return np.sign(resid)
