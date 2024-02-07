ROBOSUITE_PLANS = {
    "Lift": [
        ("red cube", "grasp")
    ],
    "Door": [
        ("door", "grasp")
    ],
    "NutAssemblyRound": [
        ("silver round nut", "grasp"), 
        ("silver peg", "place")
    ],
    "NutAssemblySquare": [
        ("gold square nut", "grasp"), 
        ("gold peg", "place")
    ],
    "NutAssembly": [
        ("gold square nut", "grasp"), 
        ("gold peg", "place"), 
        ("silver round nut", "grasp"), 
        ("silver peg", "place")
    ],
    "PickPlaceCan": [
        ("red can", "grasp"), 
        ("bin4", "place")
    ], 
    "PickPlaceCereal": [
        ("cereal box", "grasp"), 
        ("bin3", "place")
    ],
    "PickPlaceBread": [
        ("bread", "grasp"), 
        ("bin2", "place")
    ],
    "PickPlaceMilk": [
        ("milk carton", "grasp"), 
        ("bin1", "place")
    ],
    "PickPlaceCerealMilk": [
        ("milk carton", "grasp"),
        ("bin1", "place"),
        ("cereal box", "grasp"),
        ("bin2", "place")
    ],
    "PickPlaceCanBread": [
        ("red can", "grasp"),
        ("bin2", "place"),
        ("bread", "grasp"),
        ("bin1", "place")
    ]
}

MOPA_PLANS = {
    "SawyerAssemblyObstacle-v0": [
        ("empty hole", "place")
    ],
    "SawyerLiftObstacle-v0": [
        ("red can", "grasp")
    ],
    "SawyerPushObstacle-v0": [
        ("red cube", "grasp")
    ]
}

METAWORLD_PLANS = {
    "assembly-v2": [
        ("green wrench", "grasp"), 
        ("small maroon peg", "place")
    ],
    "hammer-v2": [
        ("small red hammer handle", "grasp"), 
        ("gray nail on wooden box", "place")
    ],
    "bin-picking-v2": [
        ("small green cube", "grasp"), 
        ("blue bin", "place")
    ],
    "disassemble-v2":[
        ("green wrench handle", "grasp")
    ]
}

KITCHEN_PLANS = {
    "kitchen-microwave-v0": [
        ("microwave handle", "grasp")
    ],
    "kitchen-slide-v0": [
        ("slide", "grasp")
    ],
    "kitchen-kettle-v0": [
        ("kettle", "grasp")
    ],
    "kitchen-light-v0": [
        ("light", "grasp")
    ],
    "kitchen-tlb-v0": [
        ("top left burner", "grasp")
    ],
    "microwave_kettle_light_top_left_burner": [
        ("microwave handle", "grasp"),
        ("kettle", "grasp"),
        ("light", "grasp"),
        ("top left burner", "grasp")
    ]
}
