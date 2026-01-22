"""
Progressive result tracking using JSONL format.

This module provides functionality to track evaluation results progressively,
allowing resumption from failures without losing completed work.

JSONL Format:
{"sample_id": "...", "status": "success|fail", "generated": "...", "error": "...", "timestamp": 1234567890}
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ResultTracker:
    """Track evaluation results progressively in JSONL format."""

    def __init__(self, output_path: str, model: str, provider: str, tasks: str, **kwargs):
        """
        Initialize the result tracker.

        Args:
            output_path: Base directory for output files
            model: Model name (will be sanitized for filename)
            provider: Provider name (openai, vllm-chat, etc.)
            tasks: Task name (MultiPLE, CodeElo, etc.)
            **kwargs: Additional parameters for unique filename (e.g., multiple_languages)
        """
        self.output_path = Path(output_path)
        self.model = model
        self.provider = provider
        self.tasks = tasks

        # Generate unique filename components
        model_safe = self._sanitize_for_filename(model)
        self.extra_params = "_".join([f"{k}_{v}" for k, v in sorted(kwargs.items()) if v])

        # Create filename
        timestamp = int(time.time())
        filename_parts = [f"progress", model_safe, provider, tasks, self.extra_params, str(timestamp)]
        filename = "_".join([p for p in filename_parts if p]) + ".jsonl"

        self.filepath = self.output_path / filename
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Track if file exists (continuation or new run)
        self.is_continuation = self.filepath.exists()

    @staticmethod
    def _sanitize_for_filename(name: str) -> str:
        """Sanitize a string for safe filename usage."""
        # Replace slashes and other unsafe characters
        return name.replace("/", "__").replace("\\", "__").replace(":", "--")

    def exists(self) -> bool:
        """Check if the progress file exists."""
        return self.filepath.exists()

    def get_all_entries(self) -> List[Dict]:
        """
        Read all entries from the progress file.

        Returns:
            List of dictionaries containing progress entries
        """
        if not self.exists():
            return []

        entries = []
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        return entries

    def get_failed_entries(self) -> List[Dict]:
        """
        Get all entries with status="fail".

        Returns:
            List of failed entries
        """
        return [e for e in self.get_all_entries() if e.get("status") == "fail"]

    def get_succeeded_entries(self) -> List[Dict]:
        """
        Get all entries with status="success".

        Returns:
            List of succeeded entries
        """
        return [e for e in self.get_all_entries() if e.get("status") == "success"]

    def has_failures(self) -> bool:
        """
        Check if there are any failed entries.

        Returns:
            True if there are failed entries, False otherwise
        """
        return len(self.get_failed_entries()) > 0

    def append(self, sample_id: str, status: str, generated: Optional[str] = None,
               error: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Append a new entry to the progress file.

        Args:
            sample_id: Unique identifier for the sample/task
            status: "success" or "fail"
            generated: Generated code/response (for success)
            error: Error message (for failure)
            metadata: Optional additional metadata
        """
        entry = {
            "sample_id": sample_id,
            "status": status,
            "timestamp": time.time(),
        }

        if generated is not None:
            entry["generated"] = generated

        if error is not None:
            entry["error"] = error

        if metadata:
            entry["metadata"] = metadata

        with open(self.filepath, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def update(self, sample_id: str, status: str, generated: Optional[str] = None,
               error: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Update an existing entry in-place (for retries).

        This reads the entire file, replaces the entry with matching sample_id,
        and writes back. For large files, consider a different approach.

        Args:
            sample_id: Unique identifier for the sample/task
            status: "success" or "fail"
            generated: Generated code/response (for success)
            error: Error message (for failure)
            metadata: Optional additional metadata
        """
        entries = self.get_all_entries()

        # Find and update the entry
        updated = False
        for entry in entries:
            if entry.get("sample_id") == sample_id:
                entry["status"] = status
                entry["timestamp"] = time.time()

                if generated is not None:
                    entry["generated"] = generated
                    entry.pop("error", None)  # Remove error if now success

                if error is not None:
                    entry["error"] = error
                    entry.pop("generated", None)  # Remove generated if now fail

                if metadata:
                    entry["metadata"] = metadata

                updated = True
                break

        if not updated:
            # Entry doesn't exist, append instead
            self.append(sample_id, status, generated, error, metadata)
            return

        # Write back all entries
        with open(self.filepath, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def get_succeeded_sample_ids(self) -> set:
        """
        Get set of sample IDs that have succeeded.

        Returns:
            Set of sample IDs with status="success"
        """
        return {e["sample_id"] for e in self.get_succeeded_entries()}

    def get_failed_sample_ids(self) -> set:
        """
        Get set of sample IDs that have failed.

        Returns:
            Set of sample IDs with status="fail"
        """
        return {e["sample_id"] for e in self.get_failed_entries()}

    def delete(self):
        """
        Delete the progress file.

        Called after successful completion of the running phase.
        """
        if self.exists():
            self.filepath.unlink()

    def get_summary(self) -> Dict:
        """
        Get a summary of progress.

        Returns:
            Dictionary with counts and status
        """
        entries = self.get_all_entries()
        succeeded = len([e for e in entries if e.get("status") == "success"])
        failed = len([e for e in entries if e.get("status") == "fail"])

        return {
            "total": len(entries),
            "succeeded": succeeded,
            "failed": failed,
            "file": str(self.filepath),
            "is_continuation": self.is_continuation
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (f"ResultTracker(total={summary['total']}, "
                f"succeeded={summary['succeeded']}, failed={summary['failed']}, "
                f"file={summary['file']})")
