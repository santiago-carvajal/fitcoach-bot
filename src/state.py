from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage

from src.schemas.output_schemas import DietPlan, UserProfile, WorkoutRoutine


class GraphState(TypedDict):
    messages: List[BaseMessage]
    user_profile: Optional[UserProfile]
    missing_fields: List[str]
    routine: Optional[WorkoutRoutine]
    diet: Optional[DietPlan]
    stage: str  # "collecting" | "ready_to_generate" | "done"
    safety_flag: bool
    feedback_notes: Optional[str]
    week_number: int
    previous_routine: Optional[WorkoutRoutine]
    previous_diet: Optional[DietPlan]
