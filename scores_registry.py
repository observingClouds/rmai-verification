import xskillscore as xs

REGISTRY=dict(
    RMSE = lambda ref, fc, dim : xs.rmse(ref, fc, dim=dim),
    BIAS = lambda ref, fc, dim : xs.me(ref, fc, dim=dim)
)

def compute(ref, fc, score ,dim=None):
    assert score in REGISTRY, f"The score {score} is not yet implemented."
    return REGISTRY[score](ref,fc,dim)