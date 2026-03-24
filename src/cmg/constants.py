PHASE_TO_INDEX = {"Water": 0, "Interface": 1, "Air": 2}
INDEX_TO_PHASE = {value: key for key, value in PHASE_TO_INDEX.items()}

FRAGILITY_TO_INDEX = {"fragile": 0, "robust": 1, "compliant": 2}
GEOMETRY_TO_INDEX = {
    "cylindrical": 0,
    "bowl-like": 1,
    "cup-like": 2,
    "constricted-opening": 3,
}
SURFACE_TO_INDEX = {"smooth": 0, "matte": 1}

MAIN_OBJECT_IDS = [f"OBJ{i:03d}" for i in range(1, 13)]
MECHANISM_OBJECT_IDS = [f"OBJ{i:03d}" for i in range(13, 19)]


def infer_object_pool(object_id: str) -> str:
    return "mechanism" if object_id in MECHANISM_OBJECT_IDS else "main"
