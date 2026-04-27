from enum import Enum


class Disease(str, Enum):
    FLU = "flu"
    COVID = "covid"
    RSV = "rsv"


class AggLevel(str, Enum):
    NATIONAL = "national"
    STATE = "state"
    HSA = "hsa"
    SITE = "site"
    REGION = "region"


class SourceType(str, Enum):
    NHSN = "nhsn"
    NSSP = "nssp"
    FLUSURVNET = "flusurvnet"
    ILINET = "ilinet"
