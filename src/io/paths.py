"""Output path configuration and management."""

import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CaseType(str, Enum):
    """Agreement calculation case types."""

    OVERALL = "overall"
    FIELD = "field"
    LABEL = "label"


class OutputConfig(BaseModel):
    """
    Configuration for output file paths.

    Manages the directory structure for metrics output:
    - {base_dir}/overall_agreement/
    - {base_dir}/agreement_per_field/common_{True|False}/
    - {base_dir}/agreement_per_label/common_{True|False}/
    - {base_dir}/agreement_per_field/gt_breakdown_common_{True|False}/
    - {base_dir}/agreement_per_label/gt_counts_common_{True|False}/
    """

    base_dir: str
    case: CaseType | None = None
    common: bool = False

    @classmethod
    def from_data_path(
        cls,
        data_path: str,
        case: Literal["field", "label"] | None = None,
        common: bool = False,
    ) -> "OutputConfig":
        """Create OutputConfig with base_dir derived from data_path."""
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        base_dir = f"{base_name}_metrics"

        case_type = None
        if case == "field":
            case_type = CaseType.FIELD
        elif case == "label":
            case_type = CaseType.LABEL

        return cls(base_dir=base_dir, case=case_type, common=common)

    @property
    def case_subdir(self) -> str:
        """Get the case-specific subdirectory name."""
        if self.case is None:
            return "overall_agreement"
        elif self.case == CaseType.FIELD:
            return "agreement_per_field"
        elif self.case == CaseType.LABEL:
            return "agreement_per_label"
        return "overall_agreement"

    @property
    def output_subdir(self) -> str:
        """Get the full output subdirectory path."""
        if self.case is None:
            # Overall agreement doesn't use common suffix
            return os.path.join(self.base_dir, self.case_subdir)
        return os.path.join(self.base_dir, self.case_subdir, f"common_{self.common}")

    @property
    def gt_breakdown_subdir(self) -> str:
        """Get the ground truth breakdown subdirectory (for field case)."""
        return os.path.join(
            self.base_dir,
            "agreement_per_field",
            f"gt_breakdown_common_{self.common}",
        )

    @property
    def gt_counts_subdir(self) -> str:
        """Get the ground truth counts subdirectory (for label case)."""
        return os.path.join(
            self.base_dir,
            "agreement_per_label",
            f"gt_counts_common_{self.common}",
        )

    def ensure_dirs(self) -> None:
        """Create all necessary output directories."""
        os.makedirs(self.output_subdir, exist_ok=True)

        if self.case == CaseType.FIELD:
            os.makedirs(self.gt_breakdown_subdir, exist_ok=True)
        elif self.case == CaseType.LABEL:
            os.makedirs(self.gt_counts_subdir, exist_ok=True)

    def get_output_path(self, trader: str | None = None) -> str:
        """Get the output file path for a given trader (or Total if None)."""
        if trader is None:
            filename = "Total_agreement.csv"
        else:
            filename = f"agreement_{trader}.csv"
        return os.path.join(self.output_subdir, filename)

    def get_gt_breakdown_path(self, trader: str) -> str:
        """Get the ground truth breakdown file path for a trader."""
        filename = f"agreement_{trader}.csv"
        return os.path.join(self.gt_breakdown_subdir, filename)

    def get_gt_counts_path(self, trader: str) -> str:
        """Get the ground truth counts file path for a trader."""
        filename = f"agreement_{trader}.csv"
        return os.path.join(self.gt_counts_subdir, filename)
