"""
hospital.py — minimal hospital registry.

Just enough to support the `notify_hospital` tool used by the AI
decision engine: a registry of known hospitals, and a log of
notifications sent to them. Full hospital coordination (capacity,
specialty matching, continuously-updated ETA per requirement #12) is a
later component and will extend this module rather than replace it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Hospital:
    id: str
    name: str
    node_id: str  # intersection this hospital sits at


@dataclass
class HospitalNotification:
    hospital_id: str
    ambulance_id: str
    eta_seconds: float
    sim_time_s: float


class HospitalRegistry:
    def __init__(self) -> None:
        self.hospitals: Dict[str, Hospital] = {}
        self.notifications: List[HospitalNotification] = []

    def register(self, hospital: Hospital) -> None:
        self.hospitals[hospital.id] = hospital

    def exists(self, hospital_id: str) -> bool:
        return hospital_id in self.hospitals

    def find_by_node(self, node_id: str) -> Optional["Hospital"]:
        """Look up the hospital located at a given intersection, if any."""
        for hospital in self.hospitals.values():
            if hospital.node_id == node_id:
                return hospital
        return None

    def notify(self, hospital_id: str, ambulance_id: str, eta_seconds: float,
               sim_time_s: float) -> HospitalNotification:
        if hospital_id not in self.hospitals:
            raise ValueError(f"Unknown hospital '{hospital_id}'")
        if eta_seconds < 0:
            raise ValueError("eta_seconds must be non-negative")
        record = HospitalNotification(
            hospital_id=hospital_id,
            ambulance_id=ambulance_id,
            eta_seconds=eta_seconds,
            sim_time_s=sim_time_s,
        )
        self.notifications.append(record)
        return record

    def latest_for(self, hospital_id: str) -> HospitalNotification:
        matches = [n for n in self.notifications if n.hospital_id == hospital_id]
        if not matches:
            raise ValueError(f"No notifications for '{hospital_id}'")
        return matches[-1]
