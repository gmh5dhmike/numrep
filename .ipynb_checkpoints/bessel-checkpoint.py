#!/usr/bin/env python
# implementation of up/down recursion calculations for Bessel functions
# Usage: bessel <order>

from math import sin,cos

def up(x,l):
    j0=sin(x)/x
    if l==0: return j0
    j1=(j0-cos(x))/x
    if l==1: return j1
    # else use recusion relation
    l=l-1
    return (2*l+1)/x * up(x,l) - up(x,l-1)

def down(x,n):
    lMax=50
    j=[0]*(lMax+2)     # maximum order for downward calculation
    j[lMax+1] = j[lMax] = 1.    # start with "something"
    for k in range(lMax,0,-1):
        j[k-1] = ((2.*k + 1.)/x)*j[k] - j[k+1]   # recur. rel
    scale = ((sin(x))/x)/j[0];		         #   scale the result  
    return(j[n]*scale)


if __name__ == "__main__":
    import argparse,sys
    import numpy as np
    xmax=40.0
    xmin=0.1
    step=0.1
    order=0
    if len(sys.argv)>1: order = int(sys.argv[1])
    print(order)
    print(f'Calculating J{order}(x) for {xmin} <= x <= {xmax}');

    outfile=f'bessel{order}.dat'
    out = open(outfile, 'w')
    out.write(f'# Calculating J{order}(x) for {xmin} <= x <= {xmax}\n')
    out.write(f'x\t\t J{order}(x) [down]\t J{order}(x) [up]\n')
    
    for x in np.arange(xmin,xmax+step,step):
        out.write(f'{x:.1f}\t  {down(x,order):.8e}\t  {up(x,order):.8e}\n')
    
    print(f'data stored in {outfile}');
    out.close()
    
    
    
