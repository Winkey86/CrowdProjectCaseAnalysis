import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ANSWERS = ["yes", "no", "not_working"]


def read_items_long(input_path: str) -> pd.DataFrame:
    # Скрипт принимает либо папку dataset, либо конкретный items_long файл.
    path = Path(input_path)

    if path.is_dir():
        parquet_path = path / "items_long.parquet"
        csv_path = path / "items_long.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)

        raise FileNotFoundError(
            f"Directory {path} does not contain items_long.parquet or items_long.csv"
        )

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError("Input must be a directory, .parquet file, or .csv file")


def ensure_bool(series: pd.Series) -> pd.Series:
    # После чтения CSV boolean-колонки могут стать строками, поэтому нормализуем их явно.
    if series.dtype == bool:
        return series

    return series.map(
        lambda x: (
            True if str(x).lower() == "true"
            else False if str(x).lower() == "false"
            else pd.NA
        )
    ).astype("boolean")


def safe_share(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return num / den


def resolve_worker_control_votes(
    controls: pd.DataFrame,
    key_cols: list[str],
    answer_col: str,
    correctness_col: str,
) -> pd.DataFrame:
    """
    Builds one row per worker x control task.

    Logic:
    - Same worker + same control + same repeated answer => one valid vote.
    - Same worker + same control + different answers => conflicted vote.
    - Conflicted vote is excluded from correct_share / incorrect_share,
      but counted in n_conflicted_workers and conflict_worker_share.
    """
    rows = []
    # Контрольная задача идентифицируется workflow + url + gold_answer; worker_id добавляет голос исполнителя.
    group_cols = key_cols + ["worker_id"]

    for group_values, g in controls.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        base = dict(zip(group_cols, group_values))
        attempts = len(g)

        # Собираем все ответы одного исполнителя на один и тот же контроль.
        answers = [
            str(x)
            for x in g[answer_col].dropna().tolist()
            if str(x) != ""
        ]
        unique_answers = sorted(set(answers))

        correctness_values = []
        # Корректность тоже нормализуем: она может прийти bool из parquet или строкой из CSV.
        for value in g[correctness_col].dropna().tolist():
            if isinstance(value, bool):
                correctness_values.append(value)
            else:
                value_str = str(value).lower()
                if value_str == "true":
                    correctness_values.append(True)
                elif value_str == "false":
                    correctness_values.append(False)

        unique_correctness = sorted(set(correctness_values))

        # Если один исполнитель дал разные ответы на один контроль, такой голос считаем конфликтным.
        is_worker_conflict = (
            len(unique_answers) > 1
            or len(unique_correctness) > 1
        )

        if len(unique_answers) == 1 and not is_worker_conflict:
            resolved_answer = unique_answers[0]
        elif len(unique_answers) == 0:
            resolved_answer = None
        else:
            resolved_answer = "conflict"

        if len(unique_correctness) == 1 and not is_worker_conflict:
            resolved_is_correct = unique_correctness[0]
        else:
            resolved_is_correct = pd.NA

        # В выходной таблице сохраняем и итоговый голос, и следы повторов для аудита.
        rows.append({
            **base,
            "attempt_count": attempts,
            "unique_answers_count": len(unique_answers),
            "answers_seen": "|".join(unique_answers),
            "correctness_seen": "|".join(str(x).lower() for x in unique_correctness),
            "resolved_worker_answer": resolved_answer,
            "resolved_is_correct": resolved_is_correct,
            "is_worker_conflict": bool(is_worker_conflict),
        })

    votes = pd.DataFrame(rows)

    if not votes.empty:
        votes["resolved_is_correct"] = votes["resolved_is_correct"].astype("boolean")

    return votes


def aggregate_control_metrics(
    votes: pd.DataFrame,
    key_cols: list[str],
    min_n: int,
    difficult_incorrect_share: float,
    ambiguous_low: float,
    ambiguous_high: float,
    conflict_share_threshold: float,
) -> pd.DataFrame:
    rows = []

    for group_values, g in votes.groupby(key_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        base = dict(zip(key_cols, group_values))

        n_workers_total = int(g["worker_id"].nunique())
        n_attempts_total = int(g["attempt_count"].sum())
        n_repeated_worker_votes = int((g["attempt_count"] > 1).sum())
        n_conflicted_workers = int(g["is_worker_conflict"].sum())

        # В доли правильных/ошибочных ответов входят только разрешенные голоса без конфликтов.
        valid = g[
            (~g["is_worker_conflict"])
            & (g["resolved_is_correct"].notna())
        ].copy()

        n_valid_votes = int(len(valid))
        n_correct = int((valid["resolved_is_correct"] == True).sum())
        n_incorrect = int((valid["resolved_is_correct"] == False).sum())

        # correct_share и incorrect_share считаются среди валидных голосов, поэтому в сумме дают 1.
        correct_share = safe_share(n_correct, n_valid_votes)
        incorrect_share = safe_share(n_incorrect, n_valid_votes)
        # conflict/repeat share считаются от всех работников на контроле, это отдельные диагностические доли.
        conflict_worker_share = safe_share(n_conflicted_workers, n_workers_total)
        repeat_worker_share = safe_share(n_repeated_worker_votes, n_workers_total)

        answer_counts = {}
        for answer in DEFAULT_ANSWERS:
            # Доли конкретных ответов нужны, чтобы понимать, в какую сторону расходятся исполнители.
            answer_counts[f"{answer}_count"] = int(
                (valid["resolved_worker_answer"] == answer).sum()
            )
            answer_counts[f"{answer}_share"] = safe_share(
                answer_counts[f"{answer}_count"],
                n_valid_votes,
            )

        eligible_by_n = n_valid_votes >= min_n

        # Порог min_n защищает выводы по контролям от случайности на малом числе голосов.
        is_difficult = (
            eligible_by_n
            and incorrect_share >= difficult_incorrect_share
        )

        is_ambiguous = (
            eligible_by_n
            and ambiguous_low <= correct_share <= ambiguous_high
        )

        # Повторный конфликт показывает не сложность самого ответа, а нестабильность повторных попыток.
        is_conflicted = (
            n_repeated_worker_votes > 0
            and conflict_worker_share >= conflict_share_threshold
        )

        flags = []
        if is_difficult:
            flags.append("difficult")
        if is_ambiguous:
            flags.append("ambiguous")
        if is_conflicted:
            flags.append("conflicted_repeats")

        rows.append({
            **base,
            "n_workers_total": n_workers_total,
            "n_attempts_total": n_attempts_total,
            "n_repeated_worker_votes": n_repeated_worker_votes,
            "n_conflicted_workers": n_conflicted_workers,
            "n_valid_votes": n_valid_votes,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "correct_share": correct_share,
            "incorrect_share": incorrect_share,
            "repeat_worker_share": repeat_worker_share,
            "conflict_worker_share": conflict_worker_share,
            **answer_counts,
            "eligible_by_n": eligible_by_n,
            "is_difficult": is_difficult,
            "is_ambiguous": is_ambiguous,
            "is_conflicted_repeats": is_conflicted,
            "flag": ",".join(flags) if flags else "ok",
        })

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        return metrics

    return metrics.sort_values(
        by=["flag", "incorrect_share", "conflict_worker_share", "n_valid_votes"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def write_outputs(
    out_dir: Path,
    votes: pd.DataFrame,
    metrics: pd.DataFrame,
    min_n: int,
    difficult_incorrect_share: float,
    ambiguous_low: float,
    ambiguous_high: float,
    conflict_share_threshold: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    votes.to_csv(out_dir / "worker_control_votes.csv", index=False)
    metrics.to_csv(out_dir / "control_task_metrics.csv", index=False)

    # Отдельно сохраняем только контроли, которые требуют проверки или замены в пуле.
    problem_controls = metrics[metrics["flag"] != "ok"].copy()
    if not problem_controls.empty:
        problem_controls["ambiguity_distance"] = (
            problem_controls["correct_share"] - 0.5
        ).abs()
        problem_controls = problem_controls.sort_values(
            [
                "is_difficult",
                "is_ambiguous",
                "incorrect_share",
                "ambiguity_distance",
                "n_valid_votes",
            ],
            ascending=[False, False, False, True, False],
        )

    size_distribution = (
        # Распределение по числу голосов помогает проверить, насколько надежен контрольный пул.
        metrics.groupby("n_valid_votes", dropna=False)
        .size()
        .reset_index(name="control_tasks")
        .sort_values("n_valid_votes")
    )

    repeat_summary = (
        # Отдельная сводка по повторам показывает, много ли случаев с повторными показами контролей.
        votes.groupby(["attempt_count", "is_worker_conflict"], dropna=False)
        .size()
        .reset_index(name="worker_control_votes")
        .sort_values(["attempt_count", "is_worker_conflict"])
    )

    problem_controls.to_csv(out_dir / "problem_controls.csv", index=False)

    summary = [
        "Control task metrics summary",
        "================================",
        f"Worker-control votes: {len(votes)}",
        f"Control attempts represented: {int(votes['attempt_count'].sum()) if len(votes) else 0}",
        f"Repeated worker-control votes: {int((votes['attempt_count'] > 1).sum()) if len(votes) else 0}",
        f"Conflicted worker-control votes: {int(votes['is_worker_conflict'].sum()) if len(votes) else 0}",
        f"Unique control tasks: {len(metrics)}",
        "Key columns: workflow, url, gold_answer",
        f"Minimum valid votes: {min_n}",
        f"Difficult rule: incorrect_share >= {difficult_incorrect_share:.2f}",
        f"Ambiguous rule: correct_share in [{ambiguous_low:.2f}, {ambiguous_high:.2f}]",
        f"Repeated-conflict rule: conflict_worker_share >= {conflict_share_threshold:.2f}",
        "",
        f"Eligible controls by n_valid_votes: {int(metrics['eligible_by_n'].sum()) if len(metrics) else 0}",
        f"Difficult controls: {int(metrics['is_difficult'].sum()) if len(metrics) else 0}",
        f"Ambiguous controls: {int(metrics['is_ambiguous'].sum()) if len(metrics) else 0}",
        f"Conflicted repeated controls: {int(metrics['is_conflicted_repeats'].sum()) if len(metrics) else 0}",
        f"All flagged controls: {len(problem_controls)}",
        "",
        "Sample size distribution:",
        size_distribution.to_string(index=False),
        "",
        "Repeat/conflict distribution:",
        repeat_summary.to_string(index=False),
    ]

    (out_dir / "control_metrics_summary.txt").write_text("\n".join(summary), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find difficult and ambiguous control tasks with corrected "
            "worker-level voting and explicit repeated-answer conflict handling."
        )
    )
    parser.add_argument(
        "input",
        help="Path to dataset directory, items_long.parquet, or items_long.csv",
    )
    parser.add_argument("--out", default="control_task_metrics_corrected")
    parser.add_argument("--min-n", type=int, default=4)
    parser.add_argument("--difficult-incorrect-share", type=float, default=0.60)
    parser.add_argument("--ambiguous-low", type=float, default=0.45)
    parser.add_argument("--ambiguous-high", type=float, default=0.55)
    parser.add_argument("--conflict-share-threshold", type=float, default=0.20)
    parser.add_argument(
        "--include-non-approved",
        action="store_true",
        help="Use non-APPROVED rows too. Default uses only APPROVED rows.",
    )

    args = parser.parse_args()

    items = read_items_long(args.input)

    required_cols = {
        "worker_id",
        "workflow",
        "url",
        "worker_answer",
        "gold_answer",
        "is_control",
        "is_correct",
    }
    missing = required_cols - set(items.columns)
    if missing:
        raise ValueError(f"items_long is missing required columns: {sorted(missing)}")

    if "assignment_status" not in items.columns and not args.include_non_approved:
        raise ValueError(
            "items_long has no assignment_status column. "
            "Use --include-non-approved if you intentionally want to skip this filter."
        )

    items["is_control"] = ensure_bool(items["is_control"])
    items["is_correct"] = ensure_bool(items["is_correct"])

    # Дальше работаем только с контрольными item-ами, где есть и ответ исполнителя, и эталон.
    controls = items.copy()

    if not args.include_non_approved:
        # По умолчанию анализируем только принятые ответы, потому что они влияют на результат проекта.
        controls = controls[controls["assignment_status"] == "APPROVED"].copy()

    # Контролем считается item с gold_answer; пустые ответы и строки без проверки качества исключаем.
    controls = controls[
        (controls["is_control"] == True)
        & controls["worker_id"].notna()
        & controls["url"].notna()
        & controls["worker_answer"].notna()
        & controls["gold_answer"].notna()
        & controls["is_correct"].notna()
    ].copy()

    controls["worker_answer"] = controls["worker_answer"].astype(str)
    controls["gold_answer"] = controls["gold_answer"].astype(str)

    key_cols = ["workflow", "url", "gold_answer"]

    # Сначала схлопываем попытки до одного голоса worker x control, затем агрегируем по контролю.
    votes = resolve_worker_control_votes(
        controls=controls,
        key_cols=key_cols,
        answer_col="worker_answer",
        correctness_col="is_correct",
    )

    metrics = aggregate_control_metrics(
        votes=votes,
        key_cols=key_cols,
        min_n=args.min_n,
        difficult_incorrect_share=args.difficult_incorrect_share,
        ambiguous_low=args.ambiguous_low,
        ambiguous_high=args.ambiguous_high,
        conflict_share_threshold=args.conflict_share_threshold,
    )

    write_outputs(
        out_dir=Path(args.out),
        votes=votes,
        metrics=metrics,
        min_n=args.min_n,
        difficult_incorrect_share=args.difficult_incorrect_share,
        ambiguous_low=args.ambiguous_low,
        ambiguous_high=args.ambiguous_high,
        conflict_share_threshold=args.conflict_share_threshold,
    )

    print("Done")
    print(f"controls raw attempts used: {len(controls)}")
    print(f"worker-control votes: {len(votes)}")
    print(f"control tasks: {len(metrics)}")
    print(f"eligible controls: {int(metrics['eligible_by_n'].sum()) if len(metrics) else 0}")
    print(f"difficult controls: {int(metrics['is_difficult'].sum()) if len(metrics) else 0}")
    print(f"ambiguous controls: {int(metrics['is_ambiguous'].sum()) if len(metrics) else 0}")
    print(f"conflicted repeated controls: {int(metrics['is_conflicted_repeats'].sum()) if len(metrics) else 0}")
    print(f"output: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
