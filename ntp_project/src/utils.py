
# Utility functions (placeholder)
def hello():
    return "hello"



# # src/utils.py
# import numpy as np

# def apply_ar_scenario(preds, scenario, param):
#     """
#     Apply alternative reality scenario to predicted traffic values.
#     - scenario: 'bicycle', 'bus', 'no_signals'
#     - param: parameter (percentage) e.g., 0.2 (20%)
#     Returns modified predictions (same shape).
#     """
#     preds = np.array(preds, dtype=float).flatten()
#     if scenario == 'bicycle':
#         # assume bicycles replace cars but consume small fraction of space -> reduce traffic proportionally
#         # factor: remaining traffic = (1 - adoption * 0.6)
#         factor = 1 - (param * 0.6)
#     elif scenario == 'bus':
#         # buses reduce traffic more per adopter (one bus replaces many cars) -> stronger reduction
#         factor = 1 - (param * 0.7)
#     elif scenario == 'no_signals':
#         # removing signals may improve flow but increase risk; assume moderate improvement
#         factor = 0.85
#     else:
#         factor = 1.0
#     return (preds * factor).tolist()



