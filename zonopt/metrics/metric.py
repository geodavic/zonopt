from zonopt.polyotpe import Polytope
from zonopt.metrics.hausdorff import _hausdorff_distance

_global_metric = 2

class metrics:
    
    @classmethod
    def hausdorff_distance(P: Polytope, Q: Polytope):
        return _hausdorff_distance(P,Q)
