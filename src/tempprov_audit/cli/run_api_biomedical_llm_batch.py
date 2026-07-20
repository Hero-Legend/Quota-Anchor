from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tempprov_audit.cli.run_local_biomedical_llm_batch import (
    SYSTEM_PROMPT_JOINT,
    SYSTEM_PROMPT_RELATION_ONLY,
    append_jsonl,
    build_user_prompt,
    completed_keys,
    fallback_parse_prediction,
    load_few_shot_examples,
    load_jsonl,
    merge_missing_prediction_fields,
    normalize_prediction,
    parse_json_like_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API LLM baselines on exported BioKUP inputs.")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-format", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", required=True, choices=["relation_only", "joint_task"])
    parser.add_argument(
        "--prompt-mode",
        default="zero_shot",
        choices=[
            "zero_shot",
            "few_shot",
            "cot",
            "few_shot_cot",
            "few_shot_cot_strict_verify",
            "few_shot_cot_rag_verify",
        ],
    )
    parser.add_argument("--few-shot-file")
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--extra-body-json", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-valid-only",
        action="store_true",
        help="Before resuming, remove rows with parse/API errors so only valid predictions count as completed.",
    )
    parser.add_argument(
        "--resume-retryable-only",
        action="store_true",
        help="Before resuming, remove only infrastructure failures; preserve malformed model outputs for validity metrics.",
    )
    parser.add_argument("--max-request-attempts", type=int, default=1)
    parser.add_argument("--retry-initial-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--retry-http-statuses",
        default="408,409,425,429,500,502,503,504",
        help="Comma-separated HTTP statuses eligible for exponential-backoff retries.",
    )
    parser.add_argument("--raw-output-log")
    return parser.parse_args()


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("document_id", "")).strip(), str(row.get("pair_id", "")).strip()


def is_valid_prediction(row: dict[str, Any]) -> bool:
    return bool(all(row_key(row)) and str(row.get("pred_relation", "")).strip() and not row.get("parse_error"))


def is_retryable_failure(row: dict[str, Any]) -> bool:
    error = str(row.get("parse_error", "")).strip().lower()
    if not error:
        return False
    match = re.match(r"http_(\d+)", error)
    if match and int(match.group(1)) in {402, 408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return any(
        marker in error
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "remote end closed",
            "temporary failure",
            "name or service not known",
        )
    )


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_row(handle, row)
    tmp_path.replace(path)


def compact_resume_files(
    output_path: Path,
    raw_output_path: Path | None,
    *,
    retryable_only: bool = False,
) -> set[tuple[str, str]]:
    retained_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if output_path.exists():
        for row in load_jsonl(output_path):
            retain = not is_retryable_failure(row) if retryable_only else is_valid_prediction(row)
            if all(row_key(row)) and retain:
                retained_by_key[row_key(row)] = row
        rewrite_jsonl(output_path, list(retained_by_key.values()))

    if raw_output_path is not None and raw_output_path.exists():
        raw_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in load_jsonl(raw_output_path):
            key = row_key(row)
            retained = retained_by_key.get(key)
            if retained is None:
                continue
            retained_error = str(retained.get("parse_error", "")).strip()
            raw_error = str(row.get("parse_error", "")).strip()
            if (not retained_error and not raw_error) or (retained_error and raw_error == retained_error):
                raw_by_key[key] = row
        rewrite_jsonl(raw_output_path, [raw_by_key[key] for key in retained_by_key if key in raw_by_key])

    return set(retained_by_key)


def anthropic_messages_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def post_json(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    api_format: str,
) -> dict[str, Any]:
    request = urllib.request.Request(url, method="POST")
    if api_format == "anthropic":
        request.add_header("X-Api-Key", api_key)
        request.add_header("anthropic-version", "2023-06-01")
    else:
        request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Content-Type", "application/json")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with urllib.request.urlopen(request, body, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return ""


def extract_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        message = response_payload["choices"][0]["message"]
        return message if isinstance(message, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def extract_text(response_payload: dict[str, Any]) -> str:
    if isinstance(response_payload.get("content"), list):
        return "\n".join(
            str(item.get("text", ""))
            for item in response_payload["content"]
            if isinstance(item, dict) and item.get("type") == "text"
        )
    message = extract_message(response_payload)
    content = flatten_content(message.get("content", ""))
    # Reasoning models may expose chain-of-thought in a separate field. The
    # benchmark must evaluate only the final answer, not mine labels from the
    # reasoning trace when generation is truncated before the JSON response.
    return content


def extract_reasoning_text(response_payload: dict[str, Any]) -> str:
    if isinstance(response_payload.get("content"), list):
        return "\n".join(
            str(item.get("thinking", item.get("text", "")))
            for item in response_payload["content"]
            if isinstance(item, dict) and item.get("type") == "thinking"
        )
    message = extract_message(response_payload)
    reasoning = flatten_content(message.get("reasoning_content", ""))
    if reasoning:
        return reasoning
    details = message.get("reasoning_details", [])
    if isinstance(details, list):
        return flatten_content([item.get("text", "") for item in details if isinstance(item, dict)])
    return ""


def build_payload(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: dict[str, Any],
    api_format: str,
) -> dict[str, Any]:
    if api_format == "anthropic":
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        }
    else:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    payload.update(extra_body)
    return payload


def request_with_error_capture(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    api_format: str,
    *,
    max_attempts: int,
    retry_initial_seconds: float,
    retry_max_seconds: float,
    retry_http_statuses: set[int],
) -> tuple[dict[str, Any], str | None, int]:
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        retry_after: float | None = None
        try:
            return post_json(url, api_key, payload, timeout, api_format=api_format), None, attempt
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            error_payload = {"error": {"status": exc.code, "body": raw}}
            error_text = f"http_{exc.code}: {raw[:500]}"
            should_retry = exc.code in retry_http_statuses
            if exc.code in {402, 429} and any(
                marker in raw.lower()
                for marker in (
                    "insufficient balance",
                    "exceeded_current_quota",
                    "suspended due to insufficient",
                )
            ):
                should_retry = False
            try:
                retry_after = float(exc.headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
        except Exception as exc:  # noqa: BLE001
            error_payload = {"error": {"status": "client_exception", "body": str(exc)}}
            error_text = str(exc)
            should_retry = True

        if attempt >= attempts or not should_retry:
            return error_payload, error_text, attempt

        exponential = min(retry_max_seconds, retry_initial_seconds * (2 ** (attempt - 1)))
        delay = retry_after if retry_after is not None else exponential * random.uniform(0.8, 1.2)
        time.sleep(max(0.0, delay))

    raise AssertionError("request retry loop exhausted unexpectedly")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key env: {args.api_key_env}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output_log) if args.raw_output_log else None
    records = load_jsonl(input_path)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    if args.resume_valid_only and args.resume_retryable_only:
        raise SystemExit("Choose only one of --resume-valid-only and --resume-retryable-only.")
    resume = args.resume or args.resume_valid_only or args.resume_retryable_only
    if args.resume_valid_only:
        done = compact_resume_files(output_path, raw_output_path)
    elif args.resume_retryable_only:
        done = compact_resume_files(output_path, raw_output_path, retryable_only=True)
    else:
        done = completed_keys(output_path) if resume else set()
    records = [row for row in records if (str(row["document_id"]).strip(), str(row["pair_id"]).strip()) not in done]
    if not records:
        print("No pending examples to process.")
        return

    few_shot_examples = load_few_shot_examples(
        Path(args.few_shot_file) if args.few_shot_file else None,
        args.few_shot_count,
        args.phase,
    )
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else {}
    retry_http_statuses = {
        int(item.strip()) for item in args.retry_http_statuses.split(",") if item.strip()
    }
    system_prompt = SYSTEM_PROMPT_RELATION_ONLY if args.phase == "relation_only" else SYSTEM_PROMPT_JOINT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_mode = "a" if resume else "w"
    output_mode = "a" if resume else "w"
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(output_mode, encoding="utf-8") as output_handle:
        if raw_output_path is not None and not resume:
            raw_output_path.write_text("", encoding="utf-8")
        try:
            for idx, record in enumerate(records, start=1):
                user_prompt = build_user_prompt(record, args.phase, args.prompt_mode, few_shot_examples)
                payload = build_payload(
                    model_name=args.model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    extra_body=extra_body,
                    api_format=args.api_format,
                )
                started = time.perf_counter()
                api_url = anthropic_messages_url(args.api_base) if args.api_format == "anthropic" else args.api_base
                response_payload, request_error, request_attempts = request_with_error_capture(
                    api_url,
                    api_key,
                    payload,
                    args.request_timeout,
                    args.api_format,
                    max_attempts=args.max_request_attempts,
                    retry_initial_seconds=args.retry_initial_seconds,
                    retry_max_seconds=args.retry_max_seconds,
                    retry_http_statuses=retry_http_statuses,
                )
                elapsed = time.perf_counter() - started
                text = extract_text(response_payload)
                reasoning_text = extract_reasoning_text(response_payload)

                parse_error = request_error
                if not text.strip() and request_error is None:
                    parse_error = "empty_final_content"
                    parsed_payload = fallback_parse_prediction(text, record, args.phase)
                else:
                    try:
                        parsed_payload = parse_json_like_text(text)
                    except Exception as exc:  # noqa: BLE001
                        parse_error = parse_error or str(exc)
                        parsed_payload = fallback_parse_prediction(text, record, args.phase)
                    else:
                        parsed_payload = merge_missing_prediction_fields(
                            parsed_payload,
                            fallback_parse_prediction(text, record, args.phase),
                            args.phase,
                        )

                normalized = normalize_prediction(parsed_payload, record, args.phase, args.model_name, args.prompt_mode)
                normalized["provider"] = f"{args.api_format}_api"
                normalized["latency_seconds"] = round(elapsed, 6)
                normalized["request_attempts"] = request_attempts
                if parse_error:
                    normalized["parse_error"] = parse_error
                write_jsonl_row(output_handle, normalized)
                if raw_output_path is not None:
                    append_jsonl(
                        raw_output_path,
                        {
                            "document_id": record["document_id"],
                            "pair_id": record["pair_id"],
                            "provider": f"{args.api_format}_api",
                            "model_name": args.model_name,
                            "prompt_mode": args.prompt_mode,
                            "latency_seconds": round(elapsed, 6),
                            "request_attempts": request_attempts,
                            "prompt": user_prompt,
                            "raw_text": text,
                            "raw_reasoning_text": reasoning_text,
                            "raw_response": response_payload,
                            "parse_error": parse_error,
                        },
                    )
                if idx == 1 or idx % 10 == 0 or idx == len(records):
                    print(
                        json.dumps(
                            {
                                "progress": f"{idx}/{len(records)}",
                                "document_id": record["document_id"],
                                "pair_id": record["pair_id"],
                                "parse_error": bool(parse_error),
                                "latency_seconds": round(elapsed, 3),
                                "request_attempts": request_attempts,
                            }
                        ),
                        flush=True,
                    )
                if args.sleep_seconds > 0 and idx < len(records):
                    time.sleep(args.sleep_seconds)
        finally:
            pass


if __name__ == "__main__":
    main()
