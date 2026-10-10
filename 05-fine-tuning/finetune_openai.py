"""
Complete OpenAI fine-tuning pipeline.

Handles file upload, job creation, progress monitoring, job listing,
and inference with fine-tuned models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    print("Error: openai not installed. Run: pip install openai")
    sys.exit(1)


def get_client() -> OpenAI:
    """Create and return an OpenAI client with API key validation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY=your_key"
        )
    return OpenAI(api_key=api_key)


def upload_training_file(
    client: OpenAI,
    file_path: str,
    purpose: str = "fine-tune",
) -> str:
    """Upload a file for fine-tuning.

    Args:
        client: OpenAI client instance.
        file_path: Path to the JSONL training file.
        purpose: File purpose (default: 'fine-tune').

    Returns:
        The uploaded file ID.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        OpenAIError: If the upload fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")

    print(f"Uploading file: {file_path}")
    with open(file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose=purpose)

    print(f"File uploaded successfully. File ID: {uploaded.id}")
    return uploaded.id


def create_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    model: str = "gpt-4o-mini-2024-07-18",
    validation_file_id: Optional[str] = None,
    hyperparameters: Optional[dict[str, Any]] = None,
    suffix: Optional[str] = None,
) -> str:
    """Create a fine-tuning job.

    Args:
        client: OpenAI client instance.
        training_file_id: ID of the uploaded training file.
        model: Base model to fine-tune.
        validation_file_id: Optional ID of the validation file.
        hyperparameters: Optional hyperparameter overrides.
        suffix: Optional custom model name suffix.

    Returns:
        The fine-tuning job ID.
    """
    print(f"\nCreating fine-tuning job...")
    print(f"  Base model: {model}")
    print(f"  Training file: {training_file_id}")

    kwargs: dict[str, Any] = {
        "training_file": training_file_id,
        "model": model,
    }

    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
        print(f"  Validation file: {validation_file_id}")

    if hyperparameters:
        kwargs["hyperparameters"] = hyperparameters
        print(f"  Hyperparameters: {hyperparameters}")

    if suffix:
        kwargs["suffix"] = suffix
        print(f"  Suffix: {suffix}")

    job = client.fine_tuning.jobs.create(**kwargs)
    print(f"Job created. Job ID: {job.id}")
    return job.id


def monitor_job(
    client: OpenAI,
    job_id: str,
    poll_interval: int = 30,
    max_wait: int = 7200,
) -> dict[str, Any]:
    """Monitor a fine-tuning job until completion.

    Args:
        client: OpenAI client instance.
        job_id: The fine-tuning job ID.
        poll_interval: Seconds between status checks.
        max_wait: Maximum seconds to wait before timing out.

    Returns:
        Final job status dictionary.

    Raises:
        TimeoutError: If job doesn't complete within max_wait.
    """
    print(f"\nMonitoring job: {job_id}")
    print(f"Polling every {poll_interval}s (max wait: {max_wait}s)")
    print("-" * 50)

    start_time = time.time()
    last_status = ""

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(f"Job did not complete within {max_wait}s")

        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        if status != last_status:
            print(f"[{elapsed:.0f}s] Status: {status}")
            last_status = status

        if status == "succeeded":
            print(f"\nJob completed successfully!")
            print(f"  Fine-tuned model: {job.fine_tuned_model}")
            return {
                "status": status,
                "fine_tuned_model": job.fine_tuned_model,
                "job_id": job.id,
            }

        if status == "failed":
            error_msg = job.error if hasattr(job, "error") and job.error else "Unknown error"
            print(f"\nJob failed: {error_msg}")
            return {"status": status, "error": str(error_msg)}

        if status == "cancelled":
            print(f"\nJob was cancelled.")
            return {"status": status}

        # Show training metrics if available
        if hasattr(job, "trained_tokens") and job.trained_tokens:
            print(f"  Trained tokens: {job.trained_tokens}")

        time.sleep(poll_interval)


def list_jobs(
    client: OpenAI,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """List recent fine-tuning jobs.

    Args:
        client: OpenAI client instance.
        limit: Maximum number of jobs to list.

    Returns:
        List of job dictionaries.
    """
    print(f"\nListing fine-tuning jobs (limit: {limit})")
    print("-" * 60)

    response = client.fine_tuning.jobs.list(limit=limit)
    jobs = []

    for job in response.data:
        job_info = {
            "id": job.id,
            "model": job.model,
            "status": job.status,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "created_at": job.created_at,
        }
        jobs.append(job_info)

        status_icon = {
            "succeeded": "+",
            "failed": "x",
            "running": "~",
            "pending": ".",
            "cancelled": "-",
        }.get(job.status, "?")

        model_str = job.fine_tuned_model or "N/A"
        print(f"  [{status_icon}] {job.id}")
        print(f"      Model: {job.model} -> {model_str}")
        print(f"      Status: {job.status}")
        print()

    return jobs


def use_fine_tuned_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Generate a response using a fine-tuned model.

    Args:
        client: OpenAI client instance.
        model_name: The fine-tuned model name/ID.
        prompt: User prompt.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated response text.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def batch_inference(
    client: OpenAI,
    model_name: str,
    prompts: list[str],
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> list[str]:
    """Run inference on multiple prompts.

    Args:
        client: OpenAI client instance.
        model_name: The fine-tuned model name/ID.
        prompts: List of user prompts.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per response.

    Returns:
        List of generated responses.
    """
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i + 1}/{len(prompts)}...")
        try:
            response = use_fine_tuned_model(
                client, model_name, prompt, system_prompt, temperature, max_tokens
            )
            results.append(response)
        except OpenAIError as e:
            print(f"  Error on prompt {i + 1}: {e}")
            results.append(f"Error: {e}")

    return results


def delete_file(client: OpenAI, file_id: str) -> None:
    """Delete an uploaded file.

    Args:
        client: OpenAI client instance.
        file_id: ID of the file to delete.
    """
    try:
        client.files.delete(file_id=file_id)
        print(f"Deleted file: {file_id}")
    except OpenAIError as e:
        print(f"Error deleting file {file_id}: {e}")


def run_full_pipeline(
    training_file: str,
    validation_file: Optional[str] = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: Optional[str] = None,
    hyperparameters: Optional[dict[str, Any]] = None,
    test_prompt: str = "",
) -> None:
    """Run the complete fine-tuning pipeline.

    Args:
        training_file: Path to training JSONL file.
        validation_file: Optional path to validation JSONL file.
        model: Base model to fine-tune.
        suffix: Optional custom model name suffix.
        hyperparameters: Optional hyperparameter overrides.
        test_prompt: Optional prompt to test the fine-tuned model.
    """
    print("=" * 60)
    print("OpenAI Fine-Tuning Pipeline")
    print("=" * 60)

    client = get_client()

    # Upload training file
    training_file_id = upload_training_file(client, training_file)

    # Upload validation file if provided
    validation_file_id = None
    if validation_file:
        validation_file_id = upload_training_file(client, validation_file)

    # Create job
    job_id = create_fine_tuning_job(
        client,
        training_file_id,
        model=model,
        validation_file_id=validation_file_id,
        hyperparameters=hyperparameters,
        suffix=suffix,
    )

    # Monitor progress
    result = monitor_job(client, job_id)

    if result["status"] == "succeeded":
        fine_tuned_model = result["fine_tuned_model"]
        print(f"\nFine-tuned model ready: {fine_tuned_model}")

        # Test inference
        if test_prompt:
            print(f"\n--- Test Inference ---")
            print(f"Prompt: {test_prompt}")
            response = use_fine_tuned_model(client, fine_tuned_model, test_prompt)
            print(f"Response: {response}")
    else:
        print(f"\nPipeline ended with status: {result['status']}")

    print("\nPipeline complete!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenAI Fine-Tuning Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a training file")
    upload_parser.add_argument("file", help="Path to JSONL file")
    upload_parser.add_argument("--purpose", default="fine-tune", help="File purpose")

    # Create job command
    create_parser = subparsers.add_parser("create", help="Create a fine-tuning job")
    create_parser.add_argument("training_file_id", help="Uploaded training file ID")
    create_parser.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="Base model")
    create_parser.add_argument("--validation-file-id", help="Validation file ID")
    create_parser.add_argument("--suffix", help="Custom model name suffix")
    create_parser.add_argument("--n-epochs", type=int, help="Number of epochs")
    create_parser.add_argument("--batch-size", type=int, help="Batch size")
    create_parser.add_argument("--learning-rate", type=float, help="Learning rate multiplier")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor a fine-tuning job")
    monitor_parser.add_argument("job_id", help="Fine-tuning job ID")
    monitor_parser.add_argument("--poll-interval", type=int, default=30, help="Poll interval seconds")

    # List command
    list_parser = subparsers.add_parser("list", help="List fine-tuning jobs")
    list_parser.add_argument("--limit", type=int, default=10, help="Max jobs to list")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Use a fine-tuned model")
    infer_parser.add_argument("model_name", help="Fine-tuned model name")
    infer_parser.add_argument("prompt", help="Prompt for inference")
    infer_parser.add_argument("--system-prompt", default="", help="System prompt")
    infer_parser.add_argument("--temperature", type=float, default=0.7)

    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("training_file", help="Path to training JSONL file")
    pipeline_parser.add_argument("--validation-file", help="Path to validation JSONL file")
    pipeline_parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    pipeline_parser.add_argument("--suffix", help="Custom model name suffix")
    pipeline_parser.add_argument("--test-prompt", default="", help="Test prompt after training")

    args = parser.parse_args()

    try:
        client = get_client()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.command == "upload":
            file_id = upload_training_file(client, args.file, args.purpose)
            print(f"\nFile ID: {file_id}")

        elif args.command == "create":
            hyperparameters = {}
            if args.n_epochs:
                hyperparameters["n_epochs"] = args.n_epochs
            if args.batch_size:
                hyperparameters["batch_size"] = args.batch_size
            if args.learning_rate:
                hyperparameters["learning_rate_multiplier"] = args.learning_rate

            job_id = create_fine_tuning_job(
                client,
                args.training_file_id,
                model=args.model,
                validation_file_id=args.validation_file_id,
                hyperparameters=hyperparameters or None,
                suffix=args.suffix,
            )
            print(f"\nJob ID: {job_id}")

        elif args.command == "monitor":
            result = monitor_job(client, args.job_id, args.poll_interval)
            print(f"\nResult: {json.dumps(result, indent=2)}")

        elif args.command == "list":
            list_jobs(client, args.limit)

        elif args.command == "infer":
            response = use_fine_tuned_model(
                client,
                args.model_name,
                args.prompt,
                system_prompt=args.system_prompt,
                temperature=args.temperature,
            )
            print(f"\nResponse: {response}")

        elif args.command == "pipeline":
            run_full_pipeline(
                training_file=args.training_file,
                validation_file=args.validation_file,
                model=args.model,
                suffix=args.suffix,
                test_prompt=args.test_prompt,
            )

        else:
            parser.print_help()

    except OpenAIError as e:
        print(f"OpenAI API Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(130)


if __name__ == "__main__":
    main()
