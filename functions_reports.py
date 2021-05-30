


class Compreport:
    """Class with
    """

    def __init__(self, wh, xmat, targets=None, geotargets=None):
        self.wh = wh
        self.xmat = xmat
        self.targets = targets
        self.geotargets = geotargets
        self.targets_init = np.dot(self.xmat.T, self.wh)
        if self.targets is not None:
            self.pdiff_init = self.targets_init / self.targets * 100 - 100

    def reweight(self,
                 method='ipopt',
                 options=None):
        # here are the results we want for every method
        # fields = ('method',
        #           'elapsed_seconds',
        #           'sspd',
        #           'wh_opt',
        #           'targets_opt',
        #           'pdiff',
        #           'g',
        #           'opts',
        #           'method_result')
        # ReweightResult = namedtuple('ReweightResult', fields, defaults=(None,) * len(fields))

        # rwres = ReweightResult(method=method,
        #                        elapsed_seconds=method_result.elapsed_seconds,
        #                        sspd=sspd,
        #                        wh_opt=method_result.wh_opt,
        #                        targets_opt=method_result.targets_opt,
        #                        pdiff=pdiff,
        #                        g=method_result.g,
        #                        opts=method_result.opts,
        #                        method_result=method_result)

        return rwres

