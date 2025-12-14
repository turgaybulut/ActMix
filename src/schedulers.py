from abc import ABC, abstractmethod
from typing import Literal


class BaseScheduler(ABC):
    @abstractmethod
    def get_value(self, current_epoch: int) -> float:
        pass


class LinearScheduler(BaseScheduler):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_epochs: int,
    ) -> None:
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_epochs = max(1, total_epochs)

    def get_value(self, current_epoch: int) -> float:
        if current_epoch >= self.total_epochs:
            return self.final_value

        progress = current_epoch / self.total_epochs
        return self.initial_value + progress * (self.final_value - self.initial_value)


class WarmupScheduler(BaseScheduler):
    def __init__(
        self,
        max_value: float,
        warmup_epochs: int,
    ) -> None:
        self.max_value = max_value
        self.warmup_epochs = max(1, warmup_epochs)

    def get_value(self, current_epoch: int) -> float:
        if current_epoch >= self.warmup_epochs:
            return self.max_value

        progress = current_epoch / self.warmup_epochs
        return progress * self.max_value


class TemperatureScheduler:
    def __init__(
        self,
        initial: float = 1.0,
        final: float = 0.01,
        anneal_epochs: int = 50,
        schedule_type: Literal["linear", "exponential"] = "linear",
    ) -> None:
        self.initial = initial
        self.final = final
        self.anneal_epochs = anneal_epochs
        self.schedule_type = schedule_type

        self._scheduler = self._create_scheduler()

    def _create_scheduler(self) -> BaseScheduler:
        if self.schedule_type == "linear":
            return LinearScheduler(self.initial, self.final, self.anneal_epochs)
        return LinearScheduler(self.initial, self.final, self.anneal_epochs)

    def get_temperature(self, current_epoch: int) -> float:
        return self._scheduler.get_value(current_epoch)


class EntropyLambdaScheduler:
    def __init__(
        self,
        lambda_max: float = 0.1,
        warmup_epochs: int = 10,
    ) -> None:
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs

        self._scheduler = WarmupScheduler(lambda_max, warmup_epochs)

    def get_lambda(self, current_epoch: int) -> float:
        return self._scheduler.get_value(current_epoch)
