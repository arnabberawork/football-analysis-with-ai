"""Core data models shared across the pipeline.

`Event` mirrors the "Event Data Schema" in Table 4 of the research proposal
(section 6.5). `Chunk` mirrors the textual, metadata-tagged units produced by
the chunking strategy in section 6.6, which form the knowledge base indexed
in the vector store.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    """A single match event, normalised from a raw StatsBomb event record."""

    match_id: str
    team_id: str
    team_name: str
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    minute: int
    second: int = 0
    period: int
    x: Optional[float] = None
    y: Optional[float] = None
    event_type: str
    possession_id: Optional[str] = None
    outcome: Optional[str] = None
    detail: Optional[str] = None
    """Free-text extra detail (e.g. pass recipient, shot body part)."""


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk, used for retrieval-time filtering."""

    match_id: str
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    minute_start: int
    minute_end: int
    period: int
    event_type: Optional[str] = None
    possession_id: Optional[str] = None
    chunk_level: str = Field(description="One of: event, time_window, sequence")


class Chunk(BaseModel):
    """A retrievable textual unit produced by the chunking pipeline."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
