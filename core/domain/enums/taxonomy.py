"""Vocabulario controlado de categorías taxonómicas."""
from enum import StrEnum

class TaxonomyCategory(StrEnum):
    ANTROPOMORFO  = "Antropomorfo"
    ZOOMORFO      = "Zoomorfo"
    GEOMETRICO    = "Geométrico"
    ASTRONOMICO   = "Astronómico"
    FITOMORFO     = "Fitomorfo"
    HIBRIDO       = "Híbrido"
    INDETERMINADO = "Indeterminado"
