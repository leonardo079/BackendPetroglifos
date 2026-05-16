"""Puerto de entrada: caso de uso principal que los adaptadores invocan."""
from abc import ABC, abstractmethod
from core.domain.value_objects.classification_result import ClassificationResult

class ClassifyPetroglyphPort(ABC):
    @abstractmethod
    async def classify(self, input_data: dict) -> ClassificationResult: ...
