PHASE_TO_INDEX = {"Water": 0, "Interface": 1, "Air": 2}
INDEX_TO_PHASE = {value: key for key, value in PHASE_TO_INDEX.items()}

MEDIUM_TO_PHASE = {"water": "Water", "air": "Air"}
PHASE_TO_MEDIUM = {value: key for key, value in MEDIUM_TO_PHASE.items()}

DIRECTION_TO_INDEX = {"W2A": 0, "A2W": 1}
INDEX_TO_DIRECTION = {value: key for key, value in DIRECTION_TO_INDEX.items()}
DIRECTION_TO_MEDIA = {
    "W2A": {
        "source_medium": "water",
        "target_medium": "air",
        "reference_medium": "water",
    },
    "A2W": {
        "source_medium": "air",
        "target_medium": "water",
        "reference_medium": "air",
    },
}

FRAGILITY_TO_INDEX = {"fragile": 0, "robust": 1, "compliant": 2}
GEOMETRY_TO_INDEX = {
    "cylindrical": 0,
    "bowl-like": 1,
    "cup-like": 2,
    "constricted-opening": 3,
}
SURFACE_TO_INDEX = {"smooth": 0, "matte": 1}

FRAGILITY_V2_TO_INDEX = {"fragile": 0, "non_fragile": 1}
SHAPE_PROFILE_V2_TO_INDEX = {"open_vessel": 0, "slender_or_narrow": 1}
SURFACE_TEXTURE_V2_TO_INDEX = {"smooth": 0, "textured_or_matte": 1}

FRAGILITY_TO_V2 = {
    "fragile": "fragile",
    "robust": "non_fragile",
    "compliant": "non_fragile",
}
GEOMETRY_TO_SHAPE_PROFILE_V2 = {
    "bowl-like": "open_vessel",
    "cup-like": "open_vessel",
    "cylindrical": "slender_or_narrow",
    "constricted-opening": "slender_or_narrow",
}
SURFACE_TO_TEXTURE_V2 = {
    "smooth": "smooth",
    "matte": "textured_or_matte",
}

MAIN_OBJECT_IDS = [f"OBJ{i:03d}" for i in range(1, 13)]
MECHANISM_OBJECT_IDS = [f"OBJ{i:03d}" for i in range(13, 19)]


def infer_object_pool(object_id: str) -> str:
    return "mechanism" if object_id in MECHANISM_OBJECT_IDS else "main"
