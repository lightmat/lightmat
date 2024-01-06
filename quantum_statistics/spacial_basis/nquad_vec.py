# I just copy-pasted the code from scipy.integrate.nquad and changed the quad 
# function to quad_vec to support vectorized multidimensional integration.

from scipy.integrate import quad_vec
from functools import partial

def nquad_vec(func, ranges, args=None, opts=None, full_output=False):
    
    depth = len(ranges)
    ranges = [rng if callable(rng) else _RangeFunc(rng) for rng in ranges]
    if args is None:
        args = ()
    if opts is None:
        opts = [dict([])] * depth

    if isinstance(opts, dict):
        opts = [_OptFunc(opts)] * depth
    else:
        opts = [opt if callable(opt) else _OptFunc(opt) for opt in opts]
    return _NQuad(func, ranges, opts, full_output).integrate(*args)


class _RangeFunc:
    def __init__(self, range_):
        self.range_ = range_

    def __call__(self, *args):
        """Return stored value.

        *args needed because range_ can be float or func, and is called with
        variable number of parameters.
        """
        return self.range_


class _OptFunc:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, *args):
        """Return stored dict."""
        return self.opt


class _NQuad:
    def __init__(self, func, ranges, opts, full_output):
        self.abserr = 0
        self.func = func
        self.ranges = ranges
        self.opts = opts
        self.maxdepth = len(ranges)
        self.full_output = full_output
        if self.full_output:
            self.out_dict = {'neval': 0}

    def integrate(self, *args, **kwargs):
        depth = kwargs.pop('depth', 0)
        if kwargs:
            raise ValueError('unexpected kwargs')

        # Get the integration range and options for this depth.
        ind = -(depth + 1)
        fn_range = self.ranges[ind]
        low, high = fn_range(*args)
        fn_opt = self.opts[ind]
        opt = dict(fn_opt(*args))

        if 'points' in opt:
            opt['points'] = [x for x in opt['points'] if low <= x <= high]
        if depth + 1 == self.maxdepth:
            f = self.func
        else:
            f = partial(self.integrate, depth=depth+1)
        quad_r = quad_vec(f, low, high, args=args, full_output=self.full_output,
                      **opt)
        value = quad_r[0]
        abserr = quad_r[1]
        if self.full_output:
            infodict = quad_r[2]
            # The 'neval' parameter in full_output returns the total
            # number of times the integrand function was evaluated.
            # Therefore, only the innermost integration loop counts.
            if depth + 1 == self.maxdepth:
                self.out_dict['neval'] += infodict['neval']
        self.abserr = max(self.abserr, abserr)
        if depth > 0:
            return value
        else:
            # Final result of N-D integration with error
            if self.full_output:
                return value, self.abserr, self.out_dict
            else:
                return value, self.abserr