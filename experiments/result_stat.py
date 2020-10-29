"""
    analyze results...
"""

import argparse
import os
import sys
import numpy
import torch
import torch.utils.data
import torchvision

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    parser.add_argument('-i', '--infile', default='', type=str,
                        metavar='PATH', help='path to the result of the test (.csv)') # see. 'test_pointlk.py'
    parser.add_argument('--val', default='#N/A', type=str,
                        metavar='VAL', help='info. for the x-axis')
    parser.add_argument('--hdr', dest='hdr', action='store_true',
                        help='outputs header line')

    args = parser.parse_args(argv)
    return args

def main(args):
    if args.hdr:
        # val: given value (for x-axis)
        # me_pos: mean error of estimated position (distance)
        # me_rad: mean error of estimated rotation (angle in radian)
        # me_twist: mean error represented as norm of twist vector
        # me_vel: translation part of the twist. (rotation part is me_rad)
        print('val, rot-mse, rot-rmse, rot-mae, trans-mse, trans-rmse, trans-mae, me_twist, me_vel')

    if args.infile:
        npdata = numpy.loadtxt(args.infile, delimiter=',', skiprows=1)  # --> (N, 12)
        res = torch.from_numpy(npdata).view(-1, 12)
        x_hat = res[:, 0:6]   # estimated twist vector
        x_mgt = -res[:, 6:12] # (minus) ground-truth

        g_hat = ptlk.se3.exp(x_hat) # [N, 4, 4], estimated matrices
        g_igt = ptlk.se3.exp(x_mgt) # [N, 4, 4], inverse of ground-truth
        dg = g_hat.bmm(g_igt) # [N, 4, 4]. if correct, dg == identity matrices.

        dp = dg[:, 0:3, 3]    # [N, 3], position error
        dx = ptlk.se3.log(dg) # [N, 6], twist error
        dw = dx[:, 0:3]       # [N, 3], rotation part of the twist error
        dv = dx[:, 3:6]       # [N, 3], translation part

        # ep = dp.norm(p=2, dim=1) # [N]
        ex = dx.norm(p=2, dim=1) # [N]
        # ew = dw.norm(p=2, dim=1) # [N]
        ev = dv.norm(p=2, dim=1) # [N]

        rot-mse = dw.norm(p=2, dim=1)**2  # [N]
        rot-mae = dw.norm(p=1, dim=1)  # [N]
        trans-mse = dp.norm(p=2, dim=1)**2  # [N]
        trans-mae = dp.norm(p=1, dim=1)  # [N]

        e = torch.stack((rot-mse, rot-mse, rot-mae, trans-mse, trans-mse, trans-mae, ex, ev)) # [8, N]
        me = torch.mean(e, dim=1) # [8]
        me[1] = torch.sqrt(me[1])
        me[4] = torch.sqrt(me[4])

        line = ','.join(map(str, me.numpy().tolist()))
        print('{},{}'.format(args.val, line))

if __name__ == '__main__':
    ARGS = options()
    main(ARGS)


#EOF