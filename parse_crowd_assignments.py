import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def parse_json_cell(value, default):
    # В TSV вложенные поля приходят строками с JSON; если ячейка битая или пустая, возвращаем default.
    if pd.isna(value) or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def get_nested(obj, path, default=None):
    # Безопасно достаем вложенное поле из JSON, чтобы отсутствие ключа не ломало весь парсинг.
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_url(url):
    # Пустые и нестроковые URL приводим к None: так их удобнее фильтровать в pandas.
    if not isinstance(url, str) or not url.strip():
        return None
    return url.strip()


def extract_domain(url):
    url = normalize_url(url)
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc.lower() or None


def extract_scheme(url):
    url = normalize_url(url)
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.scheme.lower() or None


def extract_gold_answers(task):
    # known_solutions в исходном TSV означает контрольное задание с заранее известным ответом.
    known = task.get("known_solutions") if isinstance(task, dict) else None
    if not isinstance(known, list):
        return []

    answers = []
    for solution in known:
        answer = get_nested(solution, ["output_values", "result"])
        if answer is not None:
            answers.append(answer)
    return answers


def build_items_long(df):
    rows = []

    for _, row in df.iterrows():
        assignment_id = row.get("assignment_assignment_id")
        worker_id = row.get("worker_id")
        status = row.get("assignment_status")

        solutions = parse_json_cell(row.get("assignment_raw_solutions"), [])
        tasks = parse_json_cell(row.get("task_suite_raw_tasks"), [])

        # tasks - показанные задания, solutions - ответы исполнителя; длины могут отличаться.
        if not isinstance(solutions, list):
            solutions = []
        if not isinstance(tasks, list):
            tasks = []

        max_len = max(len(tasks), len(solutions))

        for item_index in range(max_len):
            # Одна строка TSV содержит страницу заданий; разворачиваем ее до уровня одного item.
            task = tasks[item_index] if item_index < len(tasks) and isinstance(tasks[item_index], dict) else {}
            solution = solutions[item_index] if item_index < len(solutions) and isinstance(solutions[item_index], dict) else {}

            worker_answer = get_nested(solution, ["output_values", "result"])

            input_obj = get_nested(task, ["input_values", "input"], {})
            workflow = input_obj.get("workflow") if isinstance(input_obj, dict) else None
            url = get_nested(input_obj, ["view", "data", "region_markup", "url"])
            url = normalize_url(url)

            gold_answers = extract_gold_answers(task)
            is_control = len(gold_answers) > 0
            is_correct = None
            if is_control and worker_answer is not None:
                # Для обычных заданий is_correct остается пустым, потому что правильный ответ неизвестен.
                is_correct = worker_answer in gold_answers

            # items_long - главный рабочий слой: одна строка = один item внутри assignment.
            rows.append({
                "assignment_id": assignment_id,
                "worker_id": worker_id,
                "item_index": item_index,
                "assignment_status": status,
                "start_ts": row.get("assignment_start_time"),
                "submit_ts": row.get("assignment_submit_time"),
                "skip_ts": row.get("assignment_skip_time"),
                "start_dt": row.get("start_dt"),
                "submit_dt": row.get("submit_dt"),
                "skip_dt": row.get("skip_dt"),
                "duration_sec": row.get("duration_sec"),
                "workflow": workflow,
                "url": url,
                "domain": extract_domain(url),
                "scheme": extract_scheme(url),
                "worker_answer": worker_answer,
                "gold_answer": gold_answers[0] if gold_answers else None,
                "gold_answers": "|".join(map(str, gold_answers)) if gold_answers else None,
                "is_control": is_control,
                "is_correct": is_correct,
            })

    return pd.DataFrame(rows)


def build_assignments(items):
    def count_true(series):
        return series.dropna().eq(True).sum()

    # Собираем page-level таблицу: один assignment, его статус, длительность и контрольная точность.
    assignments = (
        items.groupby("assignment_id", dropna=False)
        .agg(
            worker_id=("worker_id", "first"),
            assignment_status=("assignment_status", "first"),
            start_dt=("start_dt", "first"),
            submit_dt=("submit_dt", "first"),
            skip_dt=("skip_dt", "first"),
            duration_sec=("duration_sec", "first"),
            tasks_count=("item_index", "count"),
            answers_count=("worker_answer", lambda x: x.notna().sum()),
            control_count=("is_control", count_true),
            control_answered_count=("is_correct", lambda x: x.notna().sum()),
            control_correct_count=("is_correct", count_true),
        )
        .reset_index()
    )

    assignments["control_accuracy"] = (
        assignments["control_correct_count"] / assignments["control_answered_count"]
    )
    # Если в assignment не было отвеченных контролей, точность не определена, а не равна 0.
    assignments.loc[assignments["control_answered_count"] == 0, "control_accuracy"] = pd.NA
    return assignments


def build_workers_quality(items):
    # Качество исполнителей считаем по APPROVED-ответам: только они пошли в итоговую разметку.
    approved = items[items["assignment_status"] == "APPROVED"].copy()

    if approved.empty:
        return pd.DataFrame()

    workers = (
        approved.groupby("worker_id", dropna=False)
        .agg(
            assignments_done=("assignment_id", "nunique"),
            items_done=("item_index", "count"),
            controls_done=("is_control", lambda x: x.dropna().eq(True).sum()),
            controls_correct=("is_correct", lambda x: x.dropna().eq(True).sum()),
            control_accuracy=("is_correct", lambda x: x.dropna().mean() if len(x.dropna()) else pd.NA),
            median_duration_sec=("duration_sec", "median"),
            mean_duration_sec=("duration_sec", "mean"),
            yes_count=("worker_answer", lambda x: (x == "yes").sum()),
            no_count=("worker_answer", lambda x: (x == "no").sum()),
            not_working_count=("worker_answer", lambda x: (x == "not_working").sum()),
        )
        .reset_index()
    )

    for answer in ["yes", "no", "not_working"]:
        # Доли ответов помогают быстро увидеть перекос исполнителя в одну кнопку.
        workers[f"{answer}_share"] = workers[f"{answer}_count"] / workers["items_done"]

    def segment(row):
        # Сегмент грубый и нужен только как быстрый диагностический признак.
        if row["controls_done"] < 5:
            return "unknown_low_controls"
        acc = row["control_accuracy"]
        if pd.isna(acc):
            return "unknown"
        if acc >= 0.85:
            return "good"
        if acc >= 0.65:
            return "risky"
        return "bad"

    workers["quality_segment"] = workers.apply(segment, axis=1)
    return workers


def build_url_quality(items):
    # URL-level агрегат нужен для вычислений: итоговый ответ, уверенность и расхождение голосов.
    approved = items[(items["assignment_status"] == "APPROVED") & items["url"].notna()].copy()

    if approved.empty:
        return pd.DataFrame()

    urls = (
        approved.groupby(["url", "domain"], dropna=False)
        .agg(
            shown_count=("worker_answer", "count"),
            unique_workers=("worker_id", "nunique"),
            yes_count=("worker_answer", lambda x: (x == "yes").sum()),
            no_count=("worker_answer", lambda x: (x == "no").sum()),
            not_working_count=("worker_answer", lambda x: (x == "not_working").sum()),
            control_shows=("is_control", lambda x: x.dropna().eq(True).sum()),
        )
        .reset_index()
    )

    answer_cols = ["yes_count", "no_count", "not_working_count"]
    urls["max_votes"] = urls[answer_cols].max(axis=1)
    # confidence - доля самого популярного ответа; disagreement_score растет при спорности URL.
    urls["confidence"] = urls["max_votes"] / urls["shown_count"]
    urls["disagreement_score"] = 1 - urls["confidence"]

    def final_label(row):
        votes = {
            "yes": row["yes_count"],
            "no": row["no_count"],
            "not_working": row["not_working_count"],
        }
        max_votes = max(votes.values())
        winners = [answer for answer, count in votes.items() if count == max_votes]
        # При равенстве голосов не выбираем случайный победивший ответ, а явно ставим tie.
        return winners[0] if len(winners) == 1 else "tie"

    urls["final_label"] = urls.apply(final_label, axis=1)
    urls["is_tie"] = urls["final_label"] == "tie"
    return urls.sort_values(["disagreement_score", "shown_count"], ascending=[False, False])


def write_table(df, out_dir, name, write_parquet=True):
    # CSV удобен для просмотра, parquet - для быстрых повторных расчетов без повторного JSON-парсинга.
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    if write_parquet:
        try:
            parquet_path = out_dir / f"{name}.parquet"
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:
            print(f"Parquet skipped for {name}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Parse crowdsourcing assignment TSV into analytical datasets.")
    parser.add_argument("input", help="Path to source TSV, for example part.tsv")
    parser.add_argument("--out", default="parsed", help="Output directory")
    parser.add_argument("--no-parquet", action="store_true", help="Do not write parquet files")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")

    # Переводим timestamp в даты и считаем длительность assignment для дальнейших метрик скорости.
    df["start_dt"] = pd.to_datetime(df["assignment_start_time"], unit="s", errors="coerce")
    df["submit_dt"] = pd.to_datetime(df["assignment_submit_time"], unit="s", errors="coerce")
    df["skip_dt"] = pd.to_datetime(df["assignment_skip_time"], unit="s", errors="coerce")
    df["duration_sec"] = df["assignment_submit_time"] - df["assignment_start_time"]

    items_long = build_items_long(df)
    assignments = build_assignments(items_long)
    workers_quality = build_workers_quality(items_long)
    url_quality = build_url_quality(items_long)

    # Пишем один рабочий датасет в нескольких срезах; parquet оставлен для быстрых вычислений.
    tables = {
        "items_long": items_long,
        "assignments": assignments,
        "workers_quality": workers_quality,
        "url_quality": url_quality,
    }

    for name, table in tables.items():
        write_table(table, out_dir, name, write_parquet=not args.no_parquet)

    print("Done")
    print(f"rows source: {len(df)}")
    print(f"rows items_long: {len(items_long)}")
    print(f"rows assignments: {len(assignments)}")
    print(f"rows workers_quality: {len(workers_quality)}")
    print(f"rows url_quality: {len(url_quality)}")
    print(f"output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
