
basic_features = [
        f"{feature}_{i}"
        for i in range(1, 11)
        for feature in ["P_Ask", "V_Ask", "P_Bid", "V_Bid"]
    ]

time_insensitive_features = [
 f"{feature}_{i}"
 for i in range(1, 11)
 for feature in ["Spread", "MidPrice"]
] + [
 "P_Diff_Ask", "P_Diff_Bid"
] + [
 f"{feature}_{i}"
 for i in range(1, 10)
 for feature in ["P_AbsDiffRel_Ask", "P_AbsDiffRel_Bid"]
] + [
 "P_Mean_Ask",
 "P_Mean_Bid",
 "V_Mean_Ask",
 "V_Mean_Bid"
] + [
 "P_AccDiff",
 "V_AccDiff"
]

time_sensitive_features = [
   f"{feature}_{i}"
   for i in range(1, 11)
   for feature in ["P_Deriv_Ask", "P_Deriv_Bid", "V_Deriv_Ask", "V_Deriv_Bid"]
] + [
   f"IntensityAverage_{i}"
   for i in range(1, 7)
] + [
   f"IntensityRelComparison_{i}"
   for i in range(1, 7)
] + [
   f"LimitActivityAcceleration_{i}"
   for i in range(1, 7)
]

features = basic_features + time_insensitive_features + time_sensitive_features
centralizer = {f'{i}': features[i] for i in range(144)}


def get_features_structure(level):

    level += 1

    _basic_features = [
        f"{feature}_{i}"
        for i in range(1, level)
        for feature in ["P_Ask", "V_Ask", "P_Bid", "V_Bid"]
    ]

    _time_insensitive_features = [
     f"{feature}_{i}"
     for i in range(1, level)
     for feature in ["Spread", "MidPrice"]
    ] + [
     "P_Diff_Ask", "P_Diff_Bid"
    ] + [
     f"{feature}_{i}"
     for i in range(1, level)
     for feature in ["P_AbsDiffRel_Ask", "P_AbsDiffRel_Bid"]
    ] + [
     "P_Mean_Ask",
     "P_Mean_Bid",
     "V_Mean_Ask",
     "V_Mean_Bid"
    ] + [
     "P_AccDiff",
     "V_AccDiff"
    ]

    _time_sensitive_features = [
    f"{feature}_{i}"
    for i in range(1, level)
    for feature in ["P_Deriv_Ask", "P_Deriv_Bid", "V_Deriv_Ask", "V_Deriv_Bid"]
    ] + [
    f"IntensityAverage_{i}"
    for i in range(1, 7)
    ] + [
    f"IntensityRelComparison_{i}"
    for i in range(1, 7)
    ] + [
    f"LimitActivityAcceleration_{i}"
    for i in range(1, 7)
    ]

    _features = basic_features + _time_insensitive_features + _time_sensitive_features

    _centralizer = {k: v for k, v in centralizer.items() if v in features}
    return _centralizer

