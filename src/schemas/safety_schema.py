"""
Esquemas de clasificación estructurada usados por los nodos de control del
grafo (guardrail de seguridad, lectura de feedback de progresión). Separados
de output_schemas.py porque estos no son "salidas" que el usuario recibe,
sino señales internas que deciden el ruteo del grafo.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SafetyClassification(BaseModel):
    mentions_injury: bool = Field(
        description="El usuario menciona una lesión actual o en curso."
    )
    mentions_medical_condition: bool = Field(
        description="El usuario menciona una condición médica relevante para ejercicio/dieta."
    )
    mentions_pregnancy: bool = Field(
        description="El usuario menciona un embarazo propio o en curso."
    )
    details: Optional[str] = Field(
        default=None, description="Resumen breve de lo mencionado, si algo se marcó true."
    )


class FeedbackClassification(BaseModel):
    sentiment: Literal["too_easy", "too_hard", "logistics_change", "satisfied", "other"]
    wants_regeneration: bool = Field(
        description="El usuario quiere (o su feedback implica) que se ajuste rutina/dieta."
    )
    summary: Optional[str] = Field(
        default=None, description="Resumen breve del feedback para guardar como contexto."
    )
