#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2018 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp
from numpy import array, ndarray


options['show_progress'] = False  # disable cvxopt output


def cvxopt_matrix(M):
    if type(M) is ndarray:
        return matrix(M)
    elif type(M) is spmatrix or type(M) is matrix:
        return M
    coo = M.tocoo()
    return spmatrix(
        coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using CVXOPT <http://cvxopt.org/>.
    Parameters
    ----------
    P : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Symmetric quadratic-cost matrix.
    q : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Quadratic-cost vector.
    G : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear inequality matrix.
    h : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear inequality vector.
    A : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear equality constraint matrix.
    b : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear equality constraint vector.
    solver : string, optional
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    initvals : numpy.array, optional
        Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    CVXOPT only considers the lower entries of `P`, therefore it will use a
    wrong cost function if a non-symmetric matrix is provided.
    """
    args = [cvxopt_matrix(P), cvxopt_matrix(q)]
    if G is not None:
        args.extend([cvxopt_matrix(G), cvxopt_matrix(h)])
        if A is not None:
            args.extend([cvxopt_matrix(A), cvxopt_matrix(b)])
    sol = qp(*args, solver=solver, initvals=initvals)
    if 'optimal' not in sol['status']:
        return None
    return array(sol['x']).reshape((q.shape[0],))