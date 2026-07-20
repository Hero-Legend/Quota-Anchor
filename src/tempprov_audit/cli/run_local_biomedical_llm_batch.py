from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import BitsAndBytesConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # API runners reuse prompt utilities without local model deps.
    torch = None
    BitsAndBytesConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel
except ModuleNotFoundError:
    PeftModel = None


SYSTEM_PROMPT_RELATION_ONLY = (
    "You are an expert biomedical relation extraction assistant. "
    "Determine the single most appropriate relation label between two biomedical entities "
    "using only the provided document content. Return exactly one compact JSON object and nothing else."
)


SYSTEM_PROMPT_JOINT = (
    "You are an expert biomedical information extraction assistant. "
    "For a candidate entity pair in a biomedical document, identify the relation label, "
    "the supporting evidence sentence ids, and whether the relation is background knowledge "
    "or a novel finding. Return exactly one compact JSON object and nothing else."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local biomedical LLM on exported biomedical model inputs.")
    parser.add_argument("--model-path", required=True, help="Local HF checkpoint path or model id.")
    parser.add_argument("--model-name", required=True, help="Friendly model name written into outputs.")
    parser.add_argument("--input", required=True, help="Input JSONL exported by export_llm_examples.py")
    parser.add_argument("--output", required=True, help="Prediction JSONL path")
    parser.add_argument("--phase", required=True, choices=["relation_only", "joint_task"])
    parser.add_argument(
        "--prompt-mode",
        default="zero_shot",
        choices=[
            "zero_shot",
            "few_shot",
            "few_shot_compact",
            "cot",
            "few_shot_cot",
            "few_shot_cot_strict_verify",
            "few_shot_cot_rag_verify",
        ],
    )
    parser.add_argument("--few-shot-file", help="Optional JSONL file with demonstrations including gold labels.")
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--raw-output-log", help="Optional JSONL file storing raw generations")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value. Use 'none' to load normally and move the model to --load-device.",
    )
    parser.add_argument("--load-device", default="cuda", help="Device used when --device-map none is selected.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--adapter-path", help="Optional PEFT/LoRA adapter path loaded on top of --model-path.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for row in load_jsonl(path):
        document_id = str(row.get("document_id", "")).strip()
        pair_id = str(row.get("pair_id", "")).strip()
        pred_relation = str(row.get("pred_relation", "")).strip()
        if document_id and pair_id and pred_relation:
            done.add((document_id, pair_id))
    return done


def sentence_block(sentences: list[dict[str, Any]]) -> str:
    return "\n".join(f"{item['sentence_id']}: {item['text']}" for item in sentences)


def _normalized_terms(text: str) -> list[str]:
    return [item for item in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(item) >= 3]


def evidence_shortlist(record: dict[str, Any], top_k: int = 8) -> list[dict[str, Any]]:
    """Build a deterministic lexical shortlist for prompt-level RAG baselines."""
    head = str(record.get("head_text", ""))
    tail = str(record.get("tail_text", ""))
    head_lower = head.lower()
    tail_lower = tail.lower()
    head_terms = set(_normalized_terms(head))
    tail_terms = set(_normalized_terms(tail))
    cue_terms = {
        "associate",
        "associated",
        "association",
        "correlate",
        "correlated",
        "correlation",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "induce",
        "induced",
        "inhibit",
        "inhibited",
        "bind",
        "binding",
        "interaction",
        "risk",
        "susceptibility",
        "effect",
        "response",
        "result",
        "conclusion",
        "significant",
        "significantly",
    }

    rows: list[dict[str, Any]] = []
    sentences = record.get("sentences", [])
    for item in sentences:
        text = str(item.get("text", ""))
        lowered = text.lower()
        tokens = set(_normalized_terms(text))
        score = 0.0
        if head_lower and head_lower in lowered:
            score += 4.0
        if tail_lower and tail_lower in lowered:
            score += 4.0
        score += 1.5 * len(tokens & head_terms)
        score += 1.5 * len(tokens & tail_terms)
        if head_terms and tokens & head_terms and tail_terms and tokens & tail_terms:
            score += 3.0
        score += 0.75 * len(tokens & cue_terms)
        if re.search(r"\b(result|results|conclusion|conclusions|objective|background)\b", lowered):
            score += 0.5
        if score > 0:
            rows.append(
                {
                    "sentence_id": int(item["sentence_id"]),
                    "score": round(score, 4),
                    "text": text,
                }
            )

    rows = sorted(rows, key=lambda row: (-float(row["score"]), int(row["sentence_id"])))
    selected = rows[:top_k]
    if not selected:
        selected = [
            {"sentence_id": int(item["sentence_id"]), "score": 0.0, "text": str(item.get("text", ""))}
            for item in sentences[: min(top_k, len(sentences))]
        ]
    return selected


def evidence_shortlist_block(record: dict[str, Any], top_k: int = 8) -> str:
    rows = evidence_shortlist(record, top_k=top_k)
    return "\n".join(f"{row['sentence_id']} [score={row['score']}]: {row['text']}" for row in rows)


def build_relation_task_block(record: dict[str, Any]) -> str:
    return (
        f"Document ID: {record['document_id']}\n"
        f"Pair ID: {record['pair_id']}\n"
        f"Head entity: {record['head_text']} ({record['head_type']})\n"
        f"Tail entity: {record['tail_text']} ({record['tail_type']})\n\n"
        "Allowed relation labels:\n"
        f"{record['relation_label_set']}\n\n"
        "Document:\n"
        f"{record['document_text']}"
    )


def build_joint_task_block(record: dict[str, Any]) -> str:
    return (
        f"Document ID: {record['document_id']}\n"
        f"Pair ID: {record['pair_id']}\n"
        f"Head entity: {record['head_text']} ({record['head_type']})\n"
        f"Tail entity: {record['tail_text']} ({record['tail_type']})\n\n"
        "Allowed relation labels:\n"
        f"{record['relation_label_set']}\n\n"
        "Allowed novelty labels:\n"
        f"{record['novelty_label_set']}\n\n"
        "Sentence list:\n"
        f"{sentence_block(record['sentences'])}"
    )


def demonstration_json(record: dict[str, Any], phase: str) -> str:
    if phase == "relation_only":
        payload = {
            "pred_relation": record["gold_relation"],
        }
    else:
        payload = {
            "pred_relation": record["gold_relation"],
            "pred_evidence_sentence_ids": record["gold_evidence_sentence_ids"],
            "pred_novelty": record["gold_novelty"],
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def format_demonstration(record: dict[str, Any], phase: str, index: int) -> str:
    required = ["gold_relation"] if phase == "relation_only" else ["gold_relation", "gold_evidence_sentence_ids", "gold_novelty"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Few-shot record {record.get('pair_id', '<unknown>')} is missing keys: {missing}")
    task_block = build_relation_task_block(record) if phase == "relation_only" else build_joint_task_block(record)
    return f"Demonstration {index}\nCase:\n{task_block}\n\nJSON Answer:\n{demonstration_json(record, phase)}"


def format_compact_demonstration(record: dict[str, Any], phase: str, index: int) -> str:
    required = ["gold_relation"] if phase == "relation_only" else ["gold_relation", "gold_evidence_sentence_ids", "gold_novelty"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Few-shot record {record.get('pair_id', '<unknown>')} is missing keys: {missing}")
    lines = [
        f"Example {index}",
        f"Head type: {record['head_type']}",
        f"Tail type: {record['tail_type']}",
        f"Allowed relation labels: {record['relation_label_set']}",
    ]
    if phase != "relation_only":
        lines.append(f"Allowed novelty labels: {record['novelty_label_set']}")
    lines.append(f"JSON Answer: {demonstration_json(record, phase)}")
    return "\n".join(lines)


def load_few_shot_examples(path: Path | None, count: int, phase: str) -> list[dict[str, Any]]:
    if path is None or count <= 0:
        return []
    rows = load_jsonl(path)
    filtered = [row for row in rows if row.get("phase") == phase or "phase" not in row]
    return filtered[:count]


def build_user_prompt(record: dict[str, Any], phase: str, prompt_mode: str, few_shot_examples: list[dict[str, Any]]) -> str:
    task_block = build_relation_task_block(record) if phase == "relation_only" else build_joint_task_block(record)
    retrieval_block = ""
    if prompt_mode == "few_shot_cot_rag_verify" and phase == "joint_task":
        retrieval_block = (
            "\n\nRetriever-proposed evidence shortlist:\n"
            f"{evidence_shortlist_block(record)}\n\n"
            "Use this shortlist as a retrieval aid, not as ground truth. "
            "You must still verify every selected evidence id against the full sentence list above.\n"
        )
    if phase == "relation_only":
        instructions = (
            "Instructions:\n"
            "1. Predict exactly one relation label from the allowed list.\n"
            "2. Return exactly one compact JSON object with only the key pred_relation.\n"
            "3. Do not output document_id, pair_id, explanations, markdown, or extra text.\n"
        )
        output_schema = '{"pred_relation":"..."}'
    else:
        instructions = (
            "Instructions:\n"
            "1. Predict exactly one relation label from the allowed list.\n"
            "2. Output the sentence ids that directly support the predicted relation.\n"
            "3. Predict whether the relation is background or novel.\n"
            "4. Return exactly one compact JSON object with only the keys pred_relation, pred_evidence_sentence_ids, and pred_novelty.\n"
            "5. If evidence is uncertain, return an empty list [].\n"
            "6. Do not output document_id, pair_id, explanations, markdown, or extra text.\n"
        )
        output_schema = '{"pred_relation":"...","pred_evidence_sentence_ids":[0,2],"pred_novelty":"background"}'

    reasoning_instruction = ""
    if prompt_mode in {"cot", "few_shot_cot", "few_shot_cot_strict_verify", "few_shot_cot_rag_verify"}:
        reasoning_instruction = (
            "Reason step by step internally about biomedical semantics, evidence support, and novelty before returning the final answer.\n"
        )
    verification_instruction = ""
    if prompt_mode in {"few_shot_cot_strict_verify", "few_shot_cot_rag_verify"} and phase == "joint_task":
        verification_instruction = (
            "Before finalizing the JSON, internally run a strict evidence audit:\n"
            "- Check that every predicted evidence id exists in the provided sentence list.\n"
            "- Keep a sentence id only if that exact sentence directly supports the predicted relation for this head-tail pair.\n"
            "- Remove sentences that are only generic background, mention only unrelated entities, or support a different relation.\n"
            "- If the evidence is distributed across multiple sentences, include all required support sentences and no unrelated extras.\n"
            "- Do not infer evidence from biomedical common knowledge outside the provided document.\n"
        )
    if prompt_mode == "few_shot_cot_rag_verify" and phase == "joint_task":
        verification_instruction += (
            "Use the retriever shortlist to prioritize inspection, but do not output a sentence id only because it appears in the shortlist.\n"
            "If the shortlist misses a necessary support sentence, recover it from the full sentence list.\n"
        )
    final_instruction = "Return one single-line JSON object only.\n"

    few_shot_block = ""
    if prompt_mode in {"few_shot", "few_shot_compact", "few_shot_cot", "few_shot_cot_strict_verify", "few_shot_cot_rag_verify"} and few_shot_examples:
        formatter = format_compact_demonstration if prompt_mode == "few_shot_compact" else format_demonstration
        demonstrations = "\n\n".join(formatter(example, phase, idx + 1) for idx, example in enumerate(few_shot_examples))
        few_shot_block = (
            "Here are labeled demonstrations to imitate.\n"
            "Use them only as formatting and label-selection examples.\n"
            "Never copy example document ids, pair ids, or answer content.\n\n"
            f"{demonstrations}\n\n"
        )

    return (
        "Analyze the following biomedical document and entity pair.\n\n"
        f"{few_shot_block}"
        "Now solve the new unlabeled instance.\n\n"
        f"Instance:\n{task_block}{retrieval_block}\n\n"
        f"{instructions}"
        f"{reasoning_instruction}"
        f"{verification_instruction}"
        f"{final_instruction}"
        "Return JSON in the following format:\n\n"
        f"{output_schema}"
    )


def extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return stripped[first_brace : last_brace + 1]
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace : last_brace + 1]
    return stripped


def parse_json_like_text(text: str) -> dict[str, Any]:
    candidate = extract_json_block(text)
    try:
        return json.loads(candidate)
    except Exception:  # noqa: BLE001
        pass

    cleaned = candidate.replace("\\_", "_")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        pass

    pythonish = cleaned.replace("null", "None").replace("true", "True").replace("false", "False")
    parsed = ast.literal_eval(pythonish)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("Parsed JSON-like text is not a dictionary.")


def normalize_text_for_parsing(text: str) -> str:
    return text.replace("\\_", "_").strip()


def extract_string_field(text: str, keys: list[str]) -> str:
    normalized = normalize_text_for_parsing(text)
    patterns = [
        r'"{key}"\s*:\s*"([^"\n]+)"',
        r"'{key}'\s*:\s*'([^'\n]+)'",
        r'"{key}"\s*:\s*([A-Za-z][A-Za-z0-9_ -]+)',
        r"'{key}'\s*:\s*([A-Za-z][A-Za-z0-9_ -]+)",
    ]
    for key in keys:
        for pattern in patterns:
            match = re.search(pattern.format(key=re.escape(key)), normalized, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip().strip(",}]")
    return ""


def extract_relation_from_text(text: str, allowed_labels: list[str]) -> str:
    explicit = extract_string_field(text, ["pred_relation", "relation", "label"])
    if explicit:
        for label in allowed_labels:
            if explicit.lower() == label.lower():
                return label

    normalized = normalize_text_for_parsing(text)
    anchored_patterns = [
        r"relation label[^A-Za-z]+(?P<label>[A-Za-z_ -]+)",
        r"predicted relation[^A-Za-z]+(?P<label>[A-Za-z_ -]+)",
        r"the relation is[^A-Za-z]+(?P<label>[A-Za-z_ -]+)",
        r"answer[^A-Za-z]+(?P<label>[A-Za-z_ -]+)",
    ]
    for pattern in anchored_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group("label").strip().strip(".:,;!)]}")
        for label in allowed_labels:
            if candidate.lower() == label.lower():
                return label

    prompt_echo_markers = ["Allowed relation labels", "Document ID:", "Pair ID:", "Instance:", "Case:"]
    if any(marker in normalized for marker in prompt_echo_markers):
        return ""

    for label in sorted(allowed_labels, key=len, reverse=True):
        pattern = rf"(?<![A-Za-z_]){re.escape(label)}(?![A-Za-z_])"
        if re.search(pattern, normalized):
            return label
    return ""


def extract_novelty_from_text(text: str, allowed_labels: list[str]) -> str:
    explicit = extract_string_field(text, ["pred_novelty", "novelty"])
    lowered = explicit.lower() if explicit else text.lower()
    for label in allowed_labels:
        if label.lower() in lowered:
            return label
    return ""


def parse_int_list_blob(blob: str) -> list[int]:
    normalized = re.sub(r"\band\b", ",", blob, flags=re.IGNORECASE)
    return [int(item) for item in re.findall(r"\d+", normalized)]


def extract_evidence_ids_from_text(text: str) -> list[int]:
    normalized = normalize_text_for_parsing(text)
    field_matches = re.findall(
        r'"?(?:pred_evidence_sentence_ids|evidence_sentence_ids|evidence_sentences|evidence|sentences?)"?\s*:\s*(\[[^\]]*\]|[0-9,\sand]+)',
        normalized,
        flags=re.IGNORECASE,
    )
    for candidate in field_matches:
        values = parse_int_list_blob(candidate)
        if values:
            return values

    candidates = re.findall(r"\[(.*?)\]", normalized)
    for candidate in candidates:
        values = parse_int_list_blob(candidate)
        if values:
            return values
    matches = re.findall(
        r"(?:sentence|sentences|evidence|sent_id|sentence_id)[^\d]*(\d+(?:\s*(?:,|and)\s*\d+)*)",
        normalized,
        flags=re.IGNORECASE,
    )
    for candidate in matches:
        values = parse_int_list_blob(candidate)
        if values:
            return values
    return []


def fallback_parse_prediction(text: str, record: dict[str, Any], phase: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_id": record["document_id"],
        "pair_id": record["pair_id"],
        "pred_relation": extract_relation_from_text(text, record.get("relation_label_set", [])),
    }
    if phase == "joint_task":
        payload["pred_evidence_sentence_ids"] = extract_evidence_ids_from_text(text)
        payload["pred_novelty"] = extract_novelty_from_text(text, record.get("novelty_label_set", []))
    return payload


def merge_missing_prediction_fields(parsed: dict[str, Any], fallback: dict[str, Any], phase: str) -> dict[str, Any]:
    merged = dict(parsed)
    if not str(merged.get("pred_relation", "")).strip():
        merged["pred_relation"] = fallback.get("pred_relation", "")
    if phase == "joint_task":
        evidence = merged.get("pred_evidence_sentence_ids")
        if not isinstance(evidence, list) or not evidence:
            merged["pred_evidence_sentence_ids"] = fallback.get("pred_evidence_sentence_ids", [])
        if not str(merged.get("pred_novelty", "")).strip():
            merged["pred_novelty"] = fallback.get("pred_novelty", "")
    return merged


def normalize_prediction(parsed: dict[str, Any], record: dict[str, Any], phase: str, model_name: str, prompt_mode: str) -> dict[str, Any]:
    output = {
        "document_id": str(parsed.get("document_id", record["document_id"])).strip(),
        "pair_id": str(parsed.get("pair_id", record["pair_id"])).strip(),
        "pred_relation": str(parsed.get("pred_relation", "")).strip(),
        "provider": "local_biomedical_llm",
        "model_name": model_name,
        "prompt_mode": prompt_mode,
    }
    if phase == "joint_task":
        evidence = parsed.get("pred_evidence_sentence_ids", [])
        normalized_evidence: list[int] = []
        if isinstance(evidence, list):
            for item in evidence:
                try:
                    normalized_evidence.append(int(item))
                except (TypeError, ValueError):
                    continue
        output["pred_evidence_sentence_ids"] = normalized_evidence
        output["pred_novelty"] = str(parsed.get("pred_novelty", "")).strip()
    return output


def dtype_from_name(name: str) -> torch.dtype | None:
    if torch is None:
        raise RuntimeError("Local LLM execution requires torch and transformers to be installed.")
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def tokenizer_has_special_token(tokenizer: AutoTokenizer, token_text: str) -> bool:
    try:
        return tokenizer.convert_tokens_to_ids(token_text) != tokenizer.unk_token_id
    except Exception:  # noqa: BLE001
        return False


def render_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = build_messages(system_prompt, user_prompt)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            merged_user_prompt = f"{system_prompt}\n\n{user_prompt}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged_user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    if tokenizer_has_special_token(tokenizer, "<|start_header_id|>") and tokenizer_has_special_token(tokenizer, "<|eot_id|>"):
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"


def main() -> None:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise SystemExit("Local LLM execution requires torch and transformers to be installed.")

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_output_path = Path(args.raw_output_log) if args.raw_output_log else None
    records = load_jsonl(input_path)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    done = completed_keys(output_path) if args.resume else set()
    records = [row for row in records if (str(row["document_id"]).strip(), str(row["pair_id"]).strip()) not in done]
    if not records:
        print("No pending examples to process.")
        return

    few_shot_examples = load_few_shot_examples(Path(args.few_shot_file) if args.few_shot_file else None, args.few_shot_count, args.phase)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    torch_dtype = dtype_from_name(args.dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.adapter_path:
        if PeftModel is None:
            raise SystemExit("Loading --adapter-path requires the peft package to be installed.")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    if args.device_map.lower() == "none":
        model = model.to(args.load_device)
    model.eval()

    system_prompt = SYSTEM_PROMPT_RELATION_ONLY if args.phase == "relation_only" else SYSTEM_PROMPT_JOINT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(records, start=1):
        user_prompt = build_user_prompt(record, args.phase, args.prompt_mode, few_shot_examples)
        prompt_text = render_prompt(tokenizer, system_prompt, user_prompt)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        if hasattr(model, "device") and str(model.device) != "meta":
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature if args.temperature > 0 else None,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

        started = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        elapsed = time.perf_counter() - started

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        parsed_payload: dict[str, Any]
        parse_error = None
        try:
            parsed_payload = parse_json_like_text(text)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
            parsed_payload = fallback_parse_prediction(text, record, args.phase)
        else:
            parsed_payload = merge_missing_prediction_fields(
                parsed_payload,
                fallback_parse_prediction(text, record, args.phase),
                args.phase,
            )

        normalized = normalize_prediction(parsed_payload, record, args.phase, args.model_name, args.prompt_mode)
        normalized["latency_seconds"] = round(elapsed, 6)
        if parse_error:
            normalized["parse_error"] = parse_error

        append_jsonl(output_path, normalized)
        append_jsonl(
            raw_output_path,
            {
                "document_id": record["document_id"],
                "pair_id": record["pair_id"],
                "provider": "local_biomedical_llm",
                "model_name": args.model_name,
                "prompt_mode": args.prompt_mode,
                "latency_seconds": round(elapsed, 6),
                "prompt": user_prompt,
                "raw_text": text,
                "parse_error": parse_error,
            },
        )

        print(
            f"[{idx}/{len(records)}] {args.model_name} {args.phase} {args.prompt_mode} "
            f"{record['pair_id']} latency={elapsed:.3f}s parse_error={bool(parse_error)}"
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
