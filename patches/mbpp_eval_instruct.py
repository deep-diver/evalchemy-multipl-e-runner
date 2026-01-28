from typing import Dict, List, Any, Generator, Optional
import json
import os
import re
import tempfile
import time
import logging
from tqdm import tqdm
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from human_eval.evaluation import evaluate_functional_correctness
from eval.task import BaseBenchmark


class MBPPBenchmark(BaseBenchmark):
    """
    MBPP (Mostly Basic Python Programming) benchmark for evaluating
    Python code generation capabilities.
    """

    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/MBPP/data",
        num_examples: int = 3,
        start_idx: int = 10,
        end_idx: int = 510,
        debug: bool = False,
        max_tokens: int = 512,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MBPP benchmark.

        Args:
            data_dir: Directory containing MBPP datasets
            max_tokens: Maximum number of tokens for generation
            num_examples: Number of examples to show in few-shot prompt
            start_idx: Start index for evaluation examples
            end_idx: End index for evaluation examples
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.num_examples = num_examples
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.debug = debug

    def format_test_example(self, question: str, tests: List[str], code: Optional[str] = None) -> str:
        """Format a single test example."""
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(question.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and format test examples from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing task_id and formatted prompt
        """
        try:
            with open(data_path, "r") as f:
                examples = [json.loads(x) for x in f]
            self.logger.info(f"Loaded {len(examples)} examples from {data_path}")

            examples_str = []
            for i in range(1, self.num_examples + 1):
                ex = examples[i]
                example_prompt = "- Example {}:\n{}".format(
                    i, self.format_test_example(ex["text"], ex["test_list"], ex["code"])
                )
                examples_str.append(example_prompt)

            eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
            if self.debug:
                eval_range = list(eval_range)[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            for i in eval_range:
                ex = examples[i]
                prompt = self.format_test_example(ex["text"], ex["test_list"])

                prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format(
                    "\n\n".join(examples_str), prompt
                )

                yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}

        except Exception as e:
            self.logger.error(f"Error reading examples: {str(e)}")
            raise

    def extract_code(self, completion: str) -> str:
        """Extract code block from model completion."""
        if not completion:
            self.logger.warning("Empty completion, returning empty string")
            return ""

        # First, try to extract from ```python``` blocks
        code_blocks = re.findall(r"```python\n(.*?)```", completion, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            return code_blocks[0]

        # Try alternative code block formats (```python ... ``` with spaces)
        code_blocks = re.findall(r"```python\s*(.*?)\s*```", completion, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            return code_blocks[0]

        # Try generic code blocks (``` ... ```)
        code_blocks = re.findall(r"```\n*(.*?)\n*```", completion, re.DOTALL)
        if code_blocks:
            return code_blocks[0]

        # Handle deepseek-r1 thinking responses - extract code after thinking tags
        # Look for code patterns like "def ", "class ", "import " that indicate actual code
        lines = completion.split('\n')

        # Skip thinking content (lines with think tags)
        code_start = 0
        for i, line in enumerate(lines):
            if '</think>' in line or '<｜end▁of▁thinking｜>' in line:
                code_start = i + 1
                break

        # Extract from the code start point
        if code_start > 0:
            potential_code = '\n'.join(lines[code_start:]).strip()
            if potential_code:
                self.logger.info(f"Extracted code from post-thinking content")
                return potential_code

        # Fallback: look for first function/class/definition
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                # Found code-like content, extract from here
                potential_code = '\n'.join(lines[i:]).strip()
                # Remove any remaining thinking tags
                potential_code = re.sub(r'<think>.*?</think>', '', potential_code, flags=re.DOTALL)
                potential_code = re.sub(r'<｜begin▁of▁thinking｜>.*?<｜end▁of▁thinking｜>', '', potential_code, flags=re.DOTALL)
                self.logger.info(f"Extracted code using fallback method")
                return potential_code.strip()

        # Last resort: return full completion with thinking tags removed
        cleaned = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL)
        cleaned = re.sub(r'<｜begin▁of▁thinking｜>.*?<｜end▁of▁thinking｜>', '', cleaned, flags=re.DOTALL)
        self.logger.warning(f"Failed to extract code block using any method, using cleaned completion")
        return cleaned.strip()

    def generate_responses(self, model: LM, result_tracker=None) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Progressive tracking strategy:
        1. Create/load a JSONL file with ALL samples marked as "fail" initially
        2. For each sample with status "fail", make API request
        3. Update entry to "success" immediately after request completes
        4. When all entries are "success", move to evaluation
        5. If any entry is "fail", stop and retry next run

        Args:
            model: Language model instance
            result_tracker: Optional ResultTracker (for backwards compatibility)

        Returns:
            Dictionary with generated responses, or None for non-primary ranks
        """
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

            # Path to the progressive tracking file (data_dir for persistence)
            progress_file = os.path.join(self.data_dir, "generated_mbpp_prev.jsonl")

            # Check for REPLACE flag - if true, delete existing progress and start fresh
            if os.environ.get("REPLACE", "false").lower() == "true":
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                    self.logger.info(f"REPLACE=true: Deleted existing progress file: {progress_file}")

            # Load original examples
            problem_file = os.path.join(self.data_dir, "mbpp.jsonl")
            original_examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Processing {len(original_examples)} examples")

            # Step 1: Create progress file with all samples as "fail" if it doesn't exist
            if not os.path.exists(progress_file):
                self.logger.info(f"Creating progress file with all {len(original_examples)} samples as 'fail'")
                with open(progress_file, "w", encoding="utf-8") as f:
                    for ex in original_examples:
                        sample_id = f"MBPP_{ex['task_id']}"
                        entry = {
                            "sample_id": sample_id,
                            "task_id": ex["task_id"],
                            "status": "fail",
                            "prompt": ex["prompt"],
                            "error": None,
                            "gpt_completion": None,
                            "generation": None,
                            "timestamp": time.time(),
                        }
                        f.write(json.dumps(entry) + "\n")
                self.logger.info(f"Created progress file: {progress_file}")

            # Load all entries from progress file
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_entries = [json.loads(line) for line in f]
            self.logger.info(f"Loaded {len(progress_entries)} entries from progress file")

            # Count successes and failures
            succeeded = [e for e in progress_entries if e["status"] == "success"]
            failed = [e for e in progress_entries if e["status"] == "fail"]
            self.logger.info(f"Progress: {len(succeeded)} succeeded, {len(failed)} failed")

            # If all succeeded, load the data and return
            if len(failed) == 0:
                self.logger.info(f"All {len(succeeded)} samples already succeeded! Loading results...")
                return self._load_all_succeeded(temp_dir_obj, progress_entries, original_examples)

            # Step 2: Process only failed samples - batch size = NUM_CONCURRENT
            self.logger.info(f"Processing {len(failed)} failed samples...")

            # Get NUM_CONCURRENT from environment or use default
            batch_size = int(os.environ.get("NUM_CONCURRENT", "4"))
            self.logger.info(f"Batch size (NUM_CONCURRENT): {batch_size}")

            for batch_start in range(0, len(failed), batch_size):
                batch_end = min(batch_start + batch_size, len(failed))
                batch_entries = failed[batch_start:batch_end]

                self.logger.info(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end} of {len(failed)}")

                try:
                    # Prepare instances for this batch
                    instances = []
                    entry_map = {}

                    for i, entry in enumerate(batch_entries):
                        sample_id = entry["sample_id"]
                        task_id = entry["task_id"]

                        inputs = self._prepare_messages([{"role": "user", "content": entry["prompt"]}], model)
                        instance = Instance(
                            "generate_until",
                            {"task_id": task_id, "prompt": entry["prompt"]},
                            (inputs, {"max_new_tokens": self.max_tokens, "do_sample": False}),
                            batch_start + i,
                        )
                        instances.append(instance)
                        entry_map[batch_start + i] = entry

                    # Step 3: Make API requests for this batch
                    outputs = self.compute(model, instances)

                    if model.rank != 0:
                        return None

                    # Update entries with results
                    for instance, output in zip(instances, outputs):
                        entry = entry_map[instance.idx]
                        sample_id = entry["sample_id"]

                        try:
                            entry["status"] = "success"
                            entry["gpt_completion"] = output
                            entry["generation"] = self.extract_code(output)
                            entry["error"] = None
                            entry["timestamp"] = time.time()

                            # Update progress file immediately
                            self._update_progress_entry(progress_file, entry)
                            self.logger.info(f"✓ {sample_id} succeeded")

                        except Exception as e:
                            entry["status"] = "fail"
                            entry["error"] = str(e)
                            entry["timestamp"] = time.time()
                            self._update_progress_entry(progress_file, entry)
                            self.logger.error(f"✗ {sample_id} failed: {str(e)}")

                    self.logger.info(f"Batch {batch_start//batch_size + 1} completed")

                except Exception as e:
                    self.logger.error(f"Batch {batch_start//batch_size + 1} failed: {str(e)}")
                    # Mark all entries in this batch as failed
                    for entry in batch_entries:
                        entry["status"] = "fail"
                        entry["error"] = f"Batch error: {str(e)}"
                        entry["timestamp"] = time.time()
                        self._update_progress_entry(progress_file, entry)
                    continue

            # Reload to check final status
            with open(progress_file, "r", encoding="utf-8") as f:
                final_entries = [json.loads(line) for line in f]

            final_succeeded = [e for e in final_entries if e["status"] == "success"]
            final_failed = [e for e in final_entries if e["status"] == "fail"]

            # Step 5: If any failures remain, stop
            if len(final_failed) > 0:
                self.logger.warning(f"=================================================")
                self.logger.warning(f"Generation incomplete: {len(final_succeeded)} succeeded, {len(final_failed)} failed")
                self.logger.warning(f"Re-run to retry failed samples.")
                self.logger.warning(f"=================================================")
                raise Exception(f"{len(final_failed)} samples failed to generate")

            # All succeeded! Prepare results for evaluation
            self.logger.info(f"All {len(final_succeeded)} samples succeeded! Preparing for evaluation...")

            # Convert progress entries to evaluation format
            all_examples = []
            for entry in final_succeeded:
                ex = {
                    "task_id": entry["task_id"],
                    "prompt": entry["prompt"],
                    "gpt_completion": entry["gpt_completion"],
                    "generation": entry["generation"],
                }
                all_examples.append(ex)

            # Save to temp dir for evaluation
            output_path = os.path.join(temp_dir, "generated_mbpp.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for ex in all_examples:
                    f.write(json.dumps(ex) + "\n")

            self.logger.info(f"Saved {len(all_examples)} examples to {output_path}")

            return {
                "temp_dir_obj": temp_dir_obj,
                "num_examples": len(all_examples),
                "total_examples": len(original_examples),
            }

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def _update_progress_entry(self, progress_file: str, updated_entry: dict):
        """Update a single entry in the progress file."""
        # Read all entries
        with open(progress_file, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        # Find and update the matching entry
        for i, entry in enumerate(entries):
            if entry["sample_id"] == updated_entry["sample_id"]:
                entries[i] = updated_entry
                break

        # Write back all entries
        with open(progress_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def _load_all_succeeded(self, temp_dir_obj, progress_entries, original_examples):
        """Load all succeeded samples for evaluation."""
        temp_dir = temp_dir_obj.name

        # Convert progress entries to evaluation format
        all_examples = []
        for entry in progress_entries:
            if entry["status"] == "success":
                ex = {
                    "task_id": entry["task_id"],
                    "prompt": entry["prompt"],
                    "gpt_completion": entry["gpt_completion"],
                    "generation": entry["generation"],
                }
                all_examples.append(ex)

        # Save to temp dir for evaluation
        output_path = os.path.join(temp_dir, "generated_mbpp.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")

        self.logger.info(f"Loaded {len(all_examples)} succeeded samples for evaluation")

        return {
            "temp_dir_obj": temp_dir_obj,
            "num_examples": len(all_examples),
            "total_examples": len(original_examples),
        }

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated code completions.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        try:
            temp_dir_obj = results["temp_dir_obj"]
            temp_dir = temp_dir_obj.name

            result = evaluate_functional_correctness(
                input_file=os.path.join(temp_dir, "generated_mbpp.jsonl"),
                tmp_dir=temp_dir,
                problem_file=os.path.join(self.data_dir, "mbpp_test.jsonl"),
                language="python",
                is_mbpp=True,
            )

            result.update(
                {
                    "num_examples": results["num_examples"],
                    "completion_rate": results["num_examples"] / results["total_examples"],
                }
            )

            temp_dir_obj.cleanup()
            return result

        except Exception as e:
            self.logger.error(f"Error in evaluate_responses: {str(e)}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            raise

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete MBPP benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting MBPP benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)

            evaluation_results.update(
                {"benchmark_version": "mbpp", "max_tokens": self.max_tokens, "num_shot": self.num_examples}
            )

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
