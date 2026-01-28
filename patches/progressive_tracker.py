"""
Generic progressive tracking for all benchmarks.

This provides a simple pattern for benchmarks to track progress per-sample:
1. Create progress file with all samples marked as "fail" initially
2. Process only failed samples
3. Update each sample to "success" immediately after completion
4. When all succeed, move to evaluation

Usage:
    from progressive_tracker import ProgressiveTracker

    tracker = ProgressiveTracker(
        data_dir="/path/to/data",
        task_name="MBPP",
        total_samples=500,
        sample_id_factory=lambda ex: f"MBPP_{ex['task_id']}",
        prompt_factory=lambda ex: ex['prompt'],
    )

    # Get failed samples to process
    failed = tracker.get_failed()

    # Process samples
    for entry in failed:
        result = api_call(entry['prompt'])
        tracker.mark_success(entry['sample_id'], result=result)

    # Check if all done
    if tracker.is_complete():
        results = tracker.load_all_results()
        evaluate(results)
"""

import json
import os
import time
from typing import Dict, List, Callable, Optional, Any
from pathlib import Path
import logging


class ProgressiveTracker:
    """
    Generic progressive tracking for benchmarks.

    Tracks per-sample progress in a JSONL file with format:
    {
        "sample_id": "...",
        "status": "success" | "fail",
        "prompt": "...",
        "result": {...},
        "error": "...",
        "timestamp": 1234567890
    }
    """

    def __init__(
        self,
        data_dir: str,
        task_name: str,
        total_samples: int,
        sample_id_factory: Callable[[Dict], str],
        prompt_factory: Callable[[Dict], str],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the progressive tracker.

        Args:
            data_dir: Directory to store progress files
            task_name: Name of the task (e.g., "MBPP", "MultiPLE")
            total_samples: Total number of samples
            sample_id_factory: Function to generate sample_id from example
            prompt_factory: Function to extract prompt from example
            logger: Optional logger instance
        """
        self.data_dir = Path(data_dir)
        self.task_name = task_name
        self.total_samples = total_samples
        self.sample_id_factory = sample_id_factory
        self.prompt_factory = prompt_factory
        self.logger = logger or logging.getLogger(__name__)

        # Progress file path
        self.progress_file = self.data_dir / f"{task_name.lower()}_progress.jsonl"

    def initialize(self, examples: List[Dict]) -> None:
        """
        Initialize progress file with all samples as "fail" if it doesn't exist.

        Args:
            examples: List of example dictionaries
        """
        if self.progress_file.exists():
            self.logger.info(f"Progress file already exists: {self.progress_file}")
            return

        self.logger.info(f"Creating progress file with {len(examples)} samples as 'fail'")
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.progress_file, "w", encoding="utf-8") as f:
            for ex in examples:
                sample_id = self.sample_id_factory(ex)
                entry = {
                    "sample_id": sample_id,
                    "status": "fail",
                    "prompt": self.prompt_factory(ex),
                    "result": None,
                    "error": None,
                    "timestamp": time.time(),
                    "metadata": {"example": ex},
                }
                f.write(json.dumps(entry) + "\n")

        self.logger.info(f"Created progress file: {self.progress_file}")

    def load_all(self) -> List[Dict]:
        """Load all entries from progress file."""
        if not self.progress_file.exists():
            return []

        with open(self.progress_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def get_succeeded(self) -> List[Dict]:
        """Get all entries with status='success'."""
        return [e for e in self.load_all() if e["status"] == "success"]

    def get_failed(self) -> List[Dict]:
        """Get all entries with status='fail'."""
        return [e for e in self.load_all() if e["status"] == "fail"]

    def get_status(self) -> Dict[str, int]:
        """Get status summary."""
        entries = self.load_all()
        return {
            "total": len(entries),
            "succeeded": len([e for e in entries if e["status"] == "success"]),
            "failed": len([e for e in entries if e["status"] == "fail"]),
        }

    def is_complete(self) -> bool:
        """Check if all samples have succeeded."""
        status = self.get_status()
        return status["failed"] == 0 and status["total"] > 0

    def mark_success(self, sample_id: str, result: Any, metadata: Optional[Dict] = None) -> None:
        """
        Mark a sample as succeeded.

        Args:
            sample_id: Unique sample identifier
            result: The result (generated code, response, etc.)
            metadata: Optional additional metadata
        """
        self._update_entry(sample_id, "success", result=result, metadata=metadata)

    def mark_fail(self, sample_id: str, error: str, metadata: Optional[Dict] = None) -> None:
        """
        Mark a sample as failed.

        Args:
            sample_id: Unique sample identifier
            error: Error message
            metadata: Optional additional metadata
        """
        self._update_entry(sample_id, "fail", error=error, metadata=metadata)

    def _update_entry(self, sample_id: str, status: str, result: Any = None,
                     error: str = None, metadata: Dict = None) -> None:
        """Update a single entry in the progress file."""
        entries = self.load_all()

        # Find and update the matching entry
        for entry in entries:
            if entry["sample_id"] == sample_id:
                entry["status"] = status
                entry["timestamp"] = time.time()

                if result is not None:
                    entry["result"] = result
                    entry.pop("error", None)

                if error is not None:
                    entry["error"] = error
                    entry.pop("result", None)

                if metadata:
                    if "metadata" not in entry:
                        entry["metadata"] = {}
                    entry["metadata"].update(metadata)

                break

        # Write back all entries
        with open(self.progress_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        self.logger.debug(f"Updated {sample_id}: status={status}")

    def load_results(self) -> List[Dict]:
        """
        Load all results for evaluation.

        Returns list of result dictionaries (only succeeded samples).
        """
        succeeded = self.get_succeeded()
        results = []
        for entry in succeeded:
            result = {
                "sample_id": entry["sample_id"],
                "prompt": entry["prompt"],
                "result": entry.get("result"),
            }
            if "metadata" in entry and "example" in entry["metadata"]:
                result["example"] = entry["metadata"]["example"]
            results.append(result)

        self.logger.info(f"Loaded {len(results)} results for evaluation")
        return results

    def delete(self) -> None:
        """Delete the progress file after successful completion."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            self.logger.info(f"Deleted progress file: {self.progress_file}")
