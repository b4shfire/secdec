#!/usr/bin/env python3
import pySecDec as psd

if __name__ == "__main__":

    li = psd.LoopIntegralFromGraph(
            internal_lines = [['m',[1,2]],['m',[1,2]]],
            external_lines = [['p',1],['p',2]],
            replacement_rules = [('p*p', 'msq'),('m*m', 'msq')]
    )

    psd.loop_package(
        name = 'bubble1L',
        loop_integral = li,
        real_parameters = ['msq'],
        requested_orders = [0],
        contour_deformation = False
    )
