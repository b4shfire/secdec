#!/usr/bin/env python3
from pySecDec.integral_interface import IntegralLibrary
import sympy as sp

# load c++ library
bubble1L = IntegralLibrary('bubble1L/bubble1L_pylink.so')

# choose integrator
bubble1L.use_Qmc(verbosity=0, fitfunction='polysingular')

# integrate
result_without_prefactor, result_prefactor, result_with_prefactor = \
    bubble1L(real_parameters=[5.0, 1.0],
          epsrel=1e-3, epsabs=1e-10, format="json", verbose=True)
values = result_with_prefactor["sums"]["bubble1L"]

# examples how to access individual orders
print('Numerical Result')
print('eps^-1:', values[(-1,)][0], '+/- (', values[(-1,)][1], ')')
print('eps^0 :', values[( 0,)][0], '+/- (', values[( 0,)][1], ')')
