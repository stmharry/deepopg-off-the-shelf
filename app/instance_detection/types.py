from enum import Enum


class InstanceDetectionV1Category(str, Enum):
    MISSING = "MISSING"
    IMPLANT = "IMPLANT"
    ROOT_REMNANTS = "ROOT_REMNANTS"
    CROWN_BRIDGE = "CROWN_BRIDGE"
    FILLING = "FILLING"
    ENDO = "ENDO"
    CARIES = "CARIES"
    PERIAPICAL_RADIOLUCENT = "PERIAPICAL_RADIOLUCENT"


EVALUATE_WHEN_MISSING_FINDINGS: list[str] = [
    InstanceDetectionV1Category.MISSING,
    InstanceDetectionV1Category.IMPLANT,
]

EVALUATE_WHEN_NONMISSING_FINDINGS: list[str] = [
    InstanceDetectionV1Category.MISSING,  # kept only for semantics, in reality we don't have negative labels
    InstanceDetectionV1Category.ROOT_REMNANTS,
    InstanceDetectionV1Category.CROWN_BRIDGE,
    InstanceDetectionV1Category.FILLING,
    InstanceDetectionV1Category.ENDO,
    InstanceDetectionV1Category.CARIES,
    InstanceDetectionV1Category.PERIAPICAL_RADIOLUCENT,
]
