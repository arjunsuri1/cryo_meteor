from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle
from typing import Literal, overload
from collections.abc import Sequence
from fsc import fsc_score
from validate import ScalarMaximizer




#From settings.py:
MAP_SAMPLING: int = 3 
TV_MAX_NUM_ITER: int = 50 # inner loop; not for iterative-tv phase retrieval
BRACKET_FOR_GOLDEN_OPTIMIZATION: tuple[float, float] = (0.0, 0.01)  # the braket can expand
TV_MAX_WEIGHT_EXPECTED = 0.1  # this value sets the threshold for a warning to the user
TV_STOP_TOLERANCE: float = 0.00000005  # inner loop; not for iterative-tv phase retrieval


def _tv_denoise_array(*, grid_array: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return grid_array
    return denoise_tv_chambolle(  # type: ignore[no-untyped-call]
        grid_array,
        weight=weight,
        eps=TV_STOP_TOLERANCE,
        max_num_iter=TV_MAX_NUM_ITER,
    )

@overload
def tv_denoise_grid(
    half_map1: np.ndarray,
    half_map2: np.ndarray,
    *,
    full_output: Literal[False],
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray: ...


@overload
def tv_denoise_grid(
    half_map1: np.ndarray,
    half_map2: np.ndarray,
    *,
    full_output: Literal[True],
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray]: ... #removed TV metadata 



def tv_denoise_grid(
    half_map1: np.ndarray,
    half_map2: np.ndarray,
    *,
    full_output: bool = False,
    weights_to_scan: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray]: #removed metadata 
    
    def fsc_objective(tv_weight: float) -> float:
        d1 = _tv_denoise_array(grid_array=half_map1, weight=tv_weight)
        d2 = _tv_denoise_array(grid_array=half_map2, weight=tv_weight)
        return fsc_score(d1, d2)
    
    maximizer = ScalarMaximizer(objective=fsc_objective)
    if weights_to_scan is not None:
        maximizer.optimize_over_explicit_values(arguments_to_scan=weights_to_scan)
    else:
        maximizer.optimize_with_golden_algorithm(bracket=BRACKET_FOR_GOLDEN_OPTIMIZATION)

    # if maximizer.argument_optimum > TV_MAX_WEIGHT_EXPECTED:
    #     np.log.warning( 
    #         "TV regularization weight much larger than expected, something probably went wrong",
    #         weight=f"{maximizer.argument_optimum:.2f}",
    #         limit=TV_MAX_WEIGHT_EXPECTED,
    #     )   

    full_map = 0.5 * (half_map1 + half_map2)          
    final_grid = _tv_denoise_array(
            grid_array=full_map,
            weight=maximizer.argument_optimum,
        )
    final_grid *= np.sum(full_map * final_grid) / np.sum(final_grid * final_grid)  
    return final_grid

