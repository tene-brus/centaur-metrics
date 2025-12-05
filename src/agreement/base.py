"""Base classes for agreement calculation."""

from abc import ABC, abstractmethod


class AgreementCalculator(ABC):
    """
    Abstract base class for agreement calculators.

    Subclasses implement specific agreement calculation strategies:
    - OverallAgreementCalculator: Single float score
    - PerFieldAgreementCalculator: Dict of field -> score
    - PerLabelAgreementCalculator: Tuple of (agreements, counts)
    """

    @abstractmethod
    def calculate(
        self,
        trades_a: list[dict],
        trades_b: list[dict],
    ) -> float | dict | tuple:
        """
        Calculate agreement between two sets of normalized trades.

        Args:
            trades_a: Normalized trades from annotator A
            trades_b: Normalized trades from annotator B

        Returns:
            Agreement score(s) - format depends on calculator type
        """
        pass

    @abstractmethod
    def similarity(self, trade_a: dict, trade_b: dict) -> float | tuple:
        """
        Calculate similarity between two individual trades.

        Args:
            trade_a: Trade from annotator A
            trade_b: Trade from annotator B

        Returns:
            Similarity score (format depends on calculator type)
        """
        pass
