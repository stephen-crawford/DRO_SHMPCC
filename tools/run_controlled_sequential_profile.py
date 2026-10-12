#!/usr/bin/env python3
"""Cold per-policy CS cache for comparable on-demand confirmation timings."""
import run_profiled_sequential_validation as implementation
from bernoulli_confidence_sequence import endpoints
original=implementation.replay

def cold_replay(*args,**kwargs):
    endpoints.cache_clear()
    return original(*args,**kwargs)

implementation.replay=cold_replay
if __name__=='__main__':implementation.main()
