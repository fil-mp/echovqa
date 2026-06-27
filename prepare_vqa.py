"""Convert EchoVQA (LLaVA conversation format) to flat VQA records.

Input:  {in_dir}/{train,test,val}.json  with conversations: [{from, value}, ...]
Output: {out_dir}/{train,test,val}_vqa.json with
        {qid, image_name, question, answer, answer_type}

answer_type is CLOSED if the answer is a single word, else OPEN.
"""
import json
import os
import argparse
from typing import Any, Dict, List, Tuple

SPLITS = ["train", "test", "val"]


def norm_text(x: Any) -> str:
    return (x or "").strip()


def normalize_role(r: Any):
    r = (r or "").strip().lower()
    if r in ("human", "user"):
        return "human"
    if r in ("gpt", "assistant"):
        return "gpt"
    return None


def validate_conversation(convs: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not convs:
        return False, "empty"
    filtered = [m for m in convs if norm_text(m.get("value")) != ""]
    if not filtered:
        return False, "all_empty"
    roles = []
    for m in filtered:
        role = normalize_role(m.get("from"))
        if role is None:
            return False, f"unknown role {m.get('from')!r}"
        roles.append(role)
    if roles[0] != "human":
        return False, f"first role {roles[0]!r}"
    for j, r in enumerate(roles):
        exp = "human" if j % 2 == 0 else "gpt"
        if r != exp:
            return False, f"break alternation at j={j}, got {r}, expected {exp}"
    return True, "ok"


def fix_alternation_keep_text(convs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    expected = "human"
    for m in convs:
        text = norm_text(m.get("value"))
        if text == "":
            continue
        role = normalize_role(m.get("from"))
        if role is None:
            continue
        if role == expected:
            m2 = dict(m)
            m2["from"] = role
            m2["value"] = text
            out.append(m2)
            expected = "gpt" if expected == "human" else "human"
    return out


def is_closed_answer(answer: str) -> bool:
    if not answer:
        return False
    return len(answer.strip().split()) == 1


def to_image_name(item: Dict[str, Any]) -> str:
    img = str(item.get("image", "")).strip()
    return img if img else "unknown_image"


def flatten_to_vqa_records(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    qid = 1
    for item in data:
        convs = item.get("conversations", [])
        image_name = to_image_name(item)
        for k in range(0, len(convs), 2):
            if k + 1 >= len(convs):
                break
            human_msg = convs[k]
            gpt_msg = convs[k + 1]
            if normalize_role(human_msg.get("from")) != "human":
                continue
            if normalize_role(gpt_msg.get("from")) != "gpt":
                continue
            question = norm_text(human_msg.get("value"))
            answer = norm_text(gpt_msg.get("value"))
            if not question or not answer:
                continue
            out.append({
                "qid": qid,
                "image_name": image_name,
                "answer": answer,
                "answer_type": "CLOSED" if is_closed_answer(answer) else "OPEN",
                "question": question,
            })
            qid += 1
    return out


def process_split(name, in_path, vqa_out_path):
    if not os.path.exists(in_path):
        return {"split": name, "error": f"Missing input file: {in_path}"}
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    dropped = 0
    for item in data:
        convs = item.get("conversations", [])
        fixed = fix_alternation_keep_text(convs)
        ok, _ = validate_conversation(fixed)
        if ok and len(fixed) >= 2:
            new_item = dict(item)
            new_item["conversations"] = fixed
            cleaned_data.append(new_item)
        else:
            dropped += 1

    vqa_records = flatten_to_vqa_records(cleaned_data)
    with open(vqa_out_path, "w", encoding="utf-8") as f:
        json.dump(vqa_records, f, ensure_ascii=False, indent=2)

    closed = sum(1 for r in vqa_records if r["answer_type"] == "CLOSED")
    return {
        "split": name, "input_count": len(data), "clean_count": len(cleaned_data),
        "dropped": dropped, "vqa_count": len(vqa_records),
        "closed": closed, "open": len(vqa_records) - closed, "vqa_path": vqa_out_path,
    }


def main():
    ap = argparse.ArgumentParser(description="Convert EchoVQA LLaVA format to flat VQA records.")
    ap.add_argument("--in-dir", required=True, help="Directory with train.json/test.json/val.json")
    ap.add_argument("--out-dir", required=True, help="Output directory for *_vqa.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in SPLITS:
        s = process_split(
            split,
            os.path.join(args.in_dir, f"{split}.json"),
            os.path.join(args.out_dir, f"{split}_vqa.json"),
        )
        if "error" in s:
            print(f"[{split}] {s['error']}")
            continue
        print(f"[{split}] in={s['input_count']} clean={s['clean_count']} "
              f"dropped={s['dropped']} rows={s['vqa_count']} "
              f"(CLOSED={s['closed']}, OPEN={s['open']}) -> {s['vqa_path']}")


if __name__ == "__main__":
    main()
