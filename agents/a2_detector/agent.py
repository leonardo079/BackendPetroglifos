"""A2 — Detector de motivos (YOLOv8)."""
from agents.base_agent import BaseAgent, AgentInput, AgentOutput

class DetectorAgent(BaseAgent):
    name = "a2_detector"

    async def run(self, input: AgentInput) -> AgentOutput:
        # Detección de bounding boxes, descripción estructurada de motivos
        raise NotImplementedError
