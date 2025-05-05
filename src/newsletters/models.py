"""Core data models."""
from datetime import datetime

# TODO: port models from NN / JJ Notes
from enum import Enum, StrEnum
from sqlmodel import SQLModel, Field as SQLField
from pydantic import BaseModel, Field, RootModel, model_validator, computed_field
from typing import Optional, Self


class TeamMember(BaseModel):
    """Team member model."""

    name: str = Field(..., description="Name of the team member")
    title: str = Field(..., description="Title of the team member")
    image_url: str = Field(..., description="Image URL of the team member")
    bio: str = Field(..., description="Bio of the team member")

    def __hash__(self):
        return hash((self.name, self.title, self.image_url, self.bio))


class KpiStatus(StrEnum):
    """KPI status enum."""

    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    NOT_STARTED = "NOT_STARTED"


class DbBase(SQLModel, table=False):
    """Base model for all models."""


class Kpi(DbBase):
    """KPI model."""

    id: Optional[int] = SQLField(default=None, primary_key=True)
    title: str
    status: KpiStatus
    owner: str
    last_updated: datetime
    data_link: str
    """Link to the data source for the KPI"""


class Ownable(BaseModel):
    """Base class for all objects that can be owned by a TeamMember."""

    title: str
    owner: TeamMember


class KeyResult(Ownable):
    """Key Result model."""

    kpis: list[Kpi] = Field(
        ..., description="List of KPIs associated with the Key Result"
    )


class FunctionalGroup(BaseModel):
    """Functional group model."""

    name: str
    description: str
    module: str
    owner: str
    team_members: list[TeamMember]
    ownables: list[Ownable]

    @computed_field  # type: ignore
    @property
    def team_member_names(self) -> list[str]:
        """Get the names of the team members."""
        return [member.name for member in self.team_members]

    @model_validator(mode="after")
    def ensure_owners_are_team_members(self) -> Self:
        """Ensure that all owners are team members."""
        for ownable in self.ownables:
            if ownable.owner not in self.team_member_names:
                raise ValueError(
                    f"Owner {ownable.owner} is not on this team: {self.team_member_names}"
                )
        return self


class KeyResultStatus(StrEnum):
    """Key Result status enum."""

    ON_TRACK = "ON_TRACK"
    BEHIND_SCHEDULE = "BEHIND_SCHEDULE"

    MISSED = "MISSED"
    MET = "MET"
    FAR_EXCEEDED = "FAR_EXCEEDED"
    ZBB = "ZBB"


class Objective(BaseModel):
    title: str

    key_results: list[KeyResult]
    """Key Results associated with the Objective"""

    parent_key_result: KeyResult
    """One key result of a parent's (manager's) objective"""
