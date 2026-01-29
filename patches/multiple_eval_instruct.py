from typing import Dict, List, Any, Optional
import json
import os
import tempfile
import time
from pathlib import Path
from tqdm import tqdm
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from multiple.evaluation import evaluate_functional_correctness
from .utils import extract_generation_code
from eval.task import BaseBenchmark
import traceback


LANUGUAGES = [
    "adb",
    "clj",
    "cpp",
    "cs",
    "dart",
    "dfy",
    "dlang",
    "elixir",
    "fs",
    "go",
    "hs",
    "java",
    "js",
    "julia",
    "lean",
    "lua",
    "luau",
    "matlab",
    "ocaml",
    "php",
    "pl",
    "python",
    "r",
    "racket",
    "ruby",
    "rs",
    "scala",
    "sh",
    "swift",
    "ts",
    "v",
]

LANGUAGE_MAP = {
    "js": "javascript",
    "java": "java",
    "python": "python",
    "cpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "hs": "haskell",
    "php": "php",
    "r": "r",
    "ruby": "ruby",
    "rs": "rust",
    "scala": "scala",
    "sh": "bash",
    "swift": "swift",
    "ts": "typescript",
    "adb": "ada",
    "clj": "clojure",
    "dart": "dart",
    "dfy": "fsharp",
    "dlang": "d",
    "elixir": "elixir",
    "fs": "fsharp",
    "julia": "julia",
    "lean": "lean",
    "lua": "lua",
    "luau": "lua",
    "matlab": "matlab",
    "ocaml": "ocaml",
    "pl": "perl",
    "racket": "racket",
    "v": "vlang",
}

DATA_DIR = "eval/chat_benchmarks/MultiPLE/data"


class MultipleBenchmark(BaseBenchmark):
    """
    Multipl-e benchmark for evaluating code generation capabilities across different languages.

    Now with progressive tracking - survives Ctrl+C and container restarts.
    """

    def __init__(
        self,
        languages: List[str] = LANUGUAGES,
        data_dir: str = DATA_DIR,
        max_tokens: int = 1024,
        num_workers: int = 10,
        timeout: float = 15,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize multipl-e benchmark.

        Args:
            languages: List of programming languages to evaluate
            data_dir: Directory containing multipl-e datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger, system_instruction)

        # Filter languages based on MULTIPLE_LANGUAGES environment variable
        multiple_languages = os.environ.get("MULTIPLE_LANGUAGES", "")
        if multiple_languages:
            # Parse comma-separated languages from environment
            env_langs = {lang.strip() for lang in multiple_languages.split(",")}
            # Filter to only include languages that are both in env and in the passed list
            self.languages = [lang for lang in languages if lang in env_langs]
            self.logger.info(f"Filtered languages to: {self.languages} (from MULTIPLE_LANGUAGES={multiple_languages})")
        else:
            self.languages = languages

        self.data_dir = data_dir
        self.max_tokens = max_tokens or 1024
        self.num_workers = num_workers
        self.timeout = timeout
        self.debug = debug
        self.system_prompt = system_instruction or "You are a helpful programming assistant designed to complete code snippets."
        self.task_prompt = """Please generate code to complete the following problem:
        ```{lang}
        {prompt}
        ```
        """

    def _get_progress_file(self, model_identifier: str = None) -> str:
        """
        Get progress file path with model and languages in filename.

        Format: multiple_progress_{model}_{lang1}_{lang2}_{lang3}.jsonl
        Example: multiple_progress_model__deepseek__deepseek-r1__java_php.jsonl

        Args:
            model_identifier: Sanitized model identifier for unique tracking

        Returns:
            Path to the progress file
        """
        # Sort languages for consistent filename
        lang_suffix = "_".join(sorted(self.languages))

        # Sanitize model identifier for filename (slashes -> double underscores)
        if model_identifier:
            model_safe = model_identifier.replace('/', '__').replace('\\', '__')
            return os.path.join(self.data_dir, f"multiple_progress_{model_safe}_{lang_suffix}.jsonl")
        else:
            # Fallback for backwards compatibility (shouldn't happen in normal use)
            return os.path.join(self.data_dir, f"multiple_progress_{lang_suffix}.jsonl")

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using progressive tracking.

        Progressive tracking strategy:
        1. Create progress file with all samples as "fail" initially
        2. For each language, load examples and track them
        3. Process only failed samples across all languages
        4. Update each sample to "success" immediately after completion
        5. When all succeed, move to evaluation

        Args:
            model: Language model instance

        Returns:
            Dictionary with generated responses, or None for non-primary ranks
        """
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

            # Extract model identifier for unique progress tracking per model
            # This allows multiple models to run on the same task without conflicts
            model_identifier = getattr(model, 'model_identifier', 'unknown')

            progress_file = self._get_progress_file(model_identifier)
            # Store for cleanup in evaluate_responses
            self._current_progress_file = progress_file
            self.logger.info(f"Progress file: {progress_file}")

            # Check for REPLACE flag - if true, delete existing progress and start fresh
            if os.environ.get("REPLACE", "false").lower() == "true":
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                    self.logger.info(f"REPLACE=true: Deleted existing progress file: {progress_file}")

            # Step 1: Collect all examples from all languages
            all_examples = []
            for lang in self.languages:
                problem_file = os.path.join(self.data_dir, f"multipl-e-{lang}.json")
                if not os.path.exists(problem_file):
                    self.logger.warning(f"Dataset file not found: {problem_file}")
                    continue

                with open(problem_file, "r", encoding="utf-8") as fr:
                    examples = json.load(fr)

                for idx, ex in enumerate(examples):
                    all_examples.append({
                        "lang": lang,
                        "example": ex,
                        "sample_id": f"Multiple_{lang}_{idx}",
                    })

            self.logger.info(f"Total samples across all languages: {len(all_examples)}")

            # Step 2: Create progress file if it doesn't exist
            if not os.path.exists(progress_file):
                self.logger.info(f"Creating progress file with {len(all_examples)} samples as 'fail'")
                with open(progress_file, "w", encoding="utf-8") as f:
                    for item in all_examples:
                        lang = item["lang"]
                        ex = item["example"]
                        prompt = ex["prompt"]
                        formatted_prompt = self.task_prompt.format(prompt=prompt, lang=LANGUAGE_MAP[lang])

                        entry = {
                            "sample_id": item["sample_id"],
                            "lang": lang,
                            "status": "fail",
                            "prompt": formatted_prompt,
                            "system_prompt": self.system_prompt,
                            "result": None,
                            "error": None,
                            "timestamp": time.time(),
                            "metadata": {"example": ex},
                        }
                        f.write(json.dumps(entry) + "\n")
                self.logger.info(f"Created progress file: {progress_file}")

            # Load progress entries
            with open(progress_file, "r", encoding="utf-8") as f:
                progress_entries = [json.loads(line) for line in f]

            succeeded = [e for e in progress_entries if e["status"] == "success"]
            failed = [e for e in progress_entries if e["status"] == "fail"]

            self.logger.info(f"Progress: {len(succeeded)} succeeded, {len(failed)} failed")

            # If all succeeded, load and return
            if len(failed) == 0:
                self.logger.info(f"All {len(succeeded)} samples already succeeded! Loading results...")
                return self._load_all_succeeded(temp_dir_obj, progress_entries)

            # Step 3: Process only failed samples - batch size = NUM_CONCURRENT
            self.logger.info(f"Processing {len(failed)} failed samples...")

            # Get NUM_CONCURRENT from environment or use default
            batch_size = int(os.environ.get("NUM_CONCURRENT", "4"))
            self.logger.info(f"Batch size (NUM_CONCURRENT): {batch_size}")

            for batch_start in range(0, len(failed), batch_size):
                batch_end = min(batch_start + batch_size, len(failed))
                batch_entries = failed[batch_start:batch_end]

                self.logger.info(f"Processing batch {batch_start//batch_size + 1}: samples {batch_start+1}-{batch_end} of {len(failed)}")

                try:
                    # Prepare instances
                    instances = []
                    entry_map = {}

                    for i, entry in enumerate(batch_entries):
                        system_prompt = entry.get("system_prompt", self.system_prompt)
                        user_prompt = entry["prompt"]

                        inputs = model.apply_chat_template([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ])

                        instance = Instance(
                            "generate_until",
                            entry["metadata"]["example"],
                            (inputs, {"max_new_tokens": self.max_tokens, "temperature": 0.0}),
                            batch_start + i,
                        )
                        instances.append(instance)
                        entry_map[batch_start + i] = entry

                    # Generate responses for batch
                    outputs = self.compute(model, instances)

                    if model.rank != 0:
                        return None

                    # Update entries with results
                    for instance, output in zip(instances, outputs):
                        entry = entry_map[instance.idx]
                        lang = entry["lang"]
                        sample_id = entry["sample_id"]

                        try:
                            ex = entry["metadata"]["example"]
                            ex_with_output = ex.copy()
                            ex_with_output["output"] = output
                            processed = extract_generation_code(ex_with_output, lang_code=lang)

                            entry["status"] = "success"
                            entry["result"] = processed
                            entry["error"] = None
                            entry["timestamp"] = time.time()

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
                    for entry in batch_entries:
                        entry["status"] = "fail"
                        entry["error"] = f"Batch error: {str(e)}"
                        entry["timestamp"] = time.time()
                        self._update_progress_entry(progress_file, entry)
                    continue

            # Reload final status
            with open(progress_file, "r", encoding="utf-8") as f:
                final_entries = [json.loads(line) for line in f]

            final_succeeded = [e for e in final_entries if e["status"] == "success"]
            final_failed = [e for e in final_entries if e["status"] == "fail"]

            if len(final_failed) > 0:
                self.logger.warning(f"=================================================")
                self.logger.warning(f"Generation incomplete: {len(final_succeeded)} succeeded, {len(final_failed)} failed")
                self.logger.warning(f"Re-run to retry failed samples.")
                self.logger.warning(f"=================================================")
                raise Exception(f"{len(final_failed)} samples failed to generate")

            # All succeeded! Prepare results for evaluation
            self.logger.info(f"All {len(final_succeeded)} samples succeeded! Preparing for evaluation...")

            # Group by language for evaluation
            results_by_lang = {}
            for entry in final_succeeded:
                lang = entry["lang"]
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(entry["result"])

            # Save per-language files for evaluation
            for lang, examples in results_by_lang.items():
                temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")
                self.logger.info(f"Saved {len(examples)} examples for {lang}")

            results = {"temp_dir_obj": temp_dir_obj}
            results.update(results_by_lang)
            return results

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def _update_progress_entry(self, progress_file: str, updated_entry: dict):
        """Update a single entry in the progress file."""
        with open(progress_file, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]

        for i, entry in enumerate(entries):
            if entry["sample_id"] == updated_entry["sample_id"]:
                entries[i] = updated_entry
                break

        with open(progress_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def _load_all_succeeded(self, temp_dir_obj, progress_entries):
        """Load all succeeded samples for evaluation."""
        temp_dir = temp_dir_obj.name

        # Group by language
        results_by_lang = {}
        for entry in progress_entries:
            if entry["status"] == "success" and entry["result"]:
                lang = entry["lang"]
                if lang not in results_by_lang:
                    results_by_lang[lang] = []
                results_by_lang[lang].append(entry["result"])

        # Save per-language files
        for lang, examples in results_by_lang.items():
            temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

        self.logger.info(f"Loaded {sum(len(v) for v in results_by_lang.values())} samples across {len(results_by_lang)} languages")

        results = {"temp_dir_obj": temp_dir_obj}
        results.update(results_by_lang)
        return results

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

        temp_dir_obj = results["temp_dir_obj"]
        temp_dir = temp_dir_obj.name

        evaluation_results = {}

        for lang in self.languages:
            try:
                problem_file = os.path.join(self.data_dir, f"multipl-e-{lang}.json")
                temp_file_path = os.path.join(temp_dir, f"generated_{lang}.jsonl")

                if not os.path.exists(temp_file_path):
                    self.logger.warning(f"Generated file not found: {temp_file_path}")
                    continue

                result = evaluate_functional_correctness(
                    input_file=temp_file_path,
                    tmp_dir=temp_dir,
                    n_workers=self.num_workers,
                    timeout=self.timeout,
                    problem_file=problem_file,
                    language=lang,
                )

                for metric, value in result.items():
                    evaluation_results[f"{lang}_{metric}"] = value

                self.logger.info(f"Completed evaluation for {lang}")

            except Exception as e:
                self.logger.error(f"Error evaluating {lang}: {str(e)}")
                traceback.print_exc()
                continue

        temp_dir_obj.cleanup()

        # Clean up progress file after successful evaluation
        # Use the stored progress file path from generate_responses
        if hasattr(self, '_current_progress_file'):
            progress_file = self._current_progress_file
            if os.path.exists(progress_file):
                os.remove(progress_file)
                self.logger.info(f"Deleted progress file after successful evaluation: {progress_file}")

        return evaluation_results
