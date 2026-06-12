import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSWER_VALUES = ["yes", "no", "not_working"]


def read_table(path: Path) -> pd.DataFrame:
    # Parquet читается быстрее, CSV оставлен как человекочитаемый fallback.
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_optional_table(*paths: Path) -> pd.DataFrame | None:
    # Некоторые таблицы можно восстановить из items_long, поэтому читаем их как optional.
    for path in paths:
        if path.exists():
            return read_table(path)
    return None


def find_table(dataset_dir: Path, name: str) -> Path:
    # Для одного имени таблицы предпочитаем parquet, если он есть, иначе берем CSV.
    parquet_path = dataset_dir / f"{name}.parquet"
    csv_path = dataset_dir / f"{name}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Cannot find {name}.parquet or {name}.csv in {dataset_dir}")


def to_bool(series: pd.Series) -> pd.Series:
    # Нормализация нужна для одинакового поведения при чтении CSV и parquet.
    if series.dtype == bool:
        return series
    return series.map(
        lambda x: (
            True if str(x).lower() == "true"
            else False if str(x).lower() == "false"
            else pd.NA
        )
    ).astype("boolean")


def share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if float(value).is_integer():
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ")


def derive_assignments(items: pd.DataFrame) -> pd.DataFrame:
    # Фолбэк на случай, если рядом с items_long нет заранее сохраненной таблицы assignments.
    assignments = items.groupby("assignment_id", dropna=False).agg(
        worker_id=("worker_id", "first"),
        assignment_status=("assignment_status", "first"),
        start_dt=("start_dt", "first"),
        submit_dt=("submit_dt", "first"),
        skip_dt=("skip_dt", "first"),
        duration_sec=("duration_sec", "first"),
        tasks_count=("item_index", "count"),
        answers_count=("worker_answer", lambda x: x.notna().sum()),
        control_count=("is_control", lambda x: (x == True).sum()),
        control_answered_count=("is_correct", lambda x: x.notna().sum()),
        control_correct_count=("is_correct", lambda x: (x == True).sum()),
    ).reset_index()

    assignments["control_accuracy"] = assignments.apply(
        lambda row: share(row["control_correct_count"], row["control_answered_count"]),
        axis=1,
    )
    assignments.loc[assignments["control_answered_count"] == 0, "control_accuracy"] = pd.NA
    return assignments


def build_worker_metrics(
    items: pd.DataFrame,
    assignments: pd.DataFrame,
    worker_votes: pd.DataFrame | None,
) -> pd.DataFrame:
    approved_items = items[
        (items["assignment_status"] == "APPROVED")
        & items["worker_answer"].notna()
    ].copy()

    # Базовые признаки работника собираем из статусов assignment-ов и распределения ответов.
    assignment_by_worker = assignments.groupby("worker_id", dropna=False).agg(
        assignments_total=("assignment_id", "nunique"),
        approved_count=("assignment_status", lambda x: (x == "APPROVED").sum()),
        expired_count=("assignment_status", lambda x: (x == "EXPIRED").sum()),
        skipped_count=("assignment_status", lambda x: (x == "SKIPPED").sum()),
        median_duration_sec=("duration_sec", "median"),
    ).reset_index()

    answer_by_worker = approved_items.groupby("worker_id", dropna=False).agg(
        # Распределение ответов показывает возможный перекос в yes/no/not_working.
        items_done=("worker_answer", "count"),
        yes_count=("worker_answer", lambda x: (x == "yes").sum()),
        no_count=("worker_answer", lambda x: (x == "no").sum()),
        not_working_count=("worker_answer", lambda x: (x == "not_working").sum()),
    ).reset_index()

    if worker_votes is not None and len(worker_votes):
        # Для контроля берем уже схлопнутые worker-control голоса, чтобы повторы не завышали вес.
        votes = worker_votes.copy()
        votes["resolved_is_correct"] = to_bool(votes["resolved_is_correct"])
        votes["is_worker_conflict"] = to_bool(votes["is_worker_conflict"])
        valid_votes = votes[
            (votes["is_worker_conflict"] != True)
            & votes["resolved_is_correct"].notna()
        ].copy()
        control_by_worker = valid_votes.groupby("worker_id", dropna=False).agg(
            controls_done=("resolved_is_correct", "count"),
            controls_correct=("resolved_is_correct", lambda x: (x == True).sum()),
        ).reset_index()
    else:
        controls = approved_items[
            (approved_items["is_control"] == True)
            & approved_items["is_correct"].notna()
        ].copy()
        control_by_worker = controls.groupby("worker_id", dropna=False).agg(
            controls_done=("is_correct", "count"),
            controls_correct=("is_correct", lambda x: (x == True).sum()),
        ).reset_index()

    workers = assignment_by_worker.merge(answer_by_worker, on="worker_id", how="left")
    workers = workers.merge(control_by_worker, on="worker_id", how="left")

    # После merge у работников без ответов/контролей появляются NaN; для счетчиков это нули.
    count_cols = [
        "items_done",
        "yes_count",
        "no_count",
        "not_working_count",
        "controls_done",
        "controls_correct",
    ]
    for col in count_cols:
        workers[col] = workers[col].fillna(0)

    workers["unfinished_share"] = workers.apply(
        # Доля незавершенных страниц показывает операционную нестабильность работника.
        lambda row: share(row["expired_count"] + row["skipped_count"], row["assignments_total"]),
        axis=1,
    )
    workers["control_accuracy"] = workers.apply(
        lambda row: share(row["controls_correct"], row["controls_done"]),
        axis=1,
    )
    for answer in ANSWER_VALUES:
        workers[f"{answer}_share"] = workers.apply(
            lambda row: share(row[f"{answer}_count"], row["items_done"]),
            axis=1,
        )
    workers["answer_dominance_share"] = workers[
        ["yes_share", "no_share", "not_working_share"]
    ].max(axis=1)

    approved_durations = assignments[
        (assignments["assignment_status"] == "APPROVED")
        & assignments["duration_sec"].notna()
    ]["duration_sec"]
    # Быстрый работник здесь означает нижние 10% по длительности среди APPROVED assignment-ов.
    fast_threshold = (
        float(approved_durations.quantile(0.10))
        if len(approved_durations)
        else 0.0
    )

    # Флаги риска не являются баном исполнителя; они выделяют кандидатов на ручную проверку.
    workers["low_control_accuracy_flag"] = (
        (workers["controls_done"] >= 4)
        & (workers["control_accuracy"] < 0.60)
    )
    workers["answer_dominance_flag"] = (
        # Флаг срабатывает только при большом перекосе ответа и слабом/недостаточном контроле качества.
        (workers["items_done"] >= 20)
        & (workers["answer_dominance_share"] >= 0.90)
        & (
            (workers["controls_done"] < 4)
            | (workers["control_accuracy"] < 0.80)
        )
    )
    workers["fast_low_quality_flag"] = (
        # Быстрота сама по себе не риск; риск появляется вместе с ошибками на контролях.
        (workers["controls_done"] >= 4)
        & workers["median_duration_sec"].notna()
        & (workers["median_duration_sec"] <= fast_threshold)
        & (workers["control_accuracy"] < 0.80)
    )
    workers["unfinished_behavior_flag"] = (
        # Частые EXPIRED/SKIPPED не портят ответы напрямую, но показывают нестабильное прохождение страниц.
        (workers["assignments_total"] >= 3)
        & (workers["unfinished_share"] >= 0.50)
    )

    flag_cols = [
        "low_control_accuracy_flag",
        "answer_dominance_flag",
        "fast_low_quality_flag",
        "unfinished_behavior_flag",
    ]
    workers["suspicious_flag"] = workers[flag_cols].any(axis=1)

    def reasons(row):
        # В reasons сохраняем человекочитаемый список сработавших признаков для отчета.
        reason_names = [
            ("low_control_accuracy_flag", "low_control_accuracy"),
            ("answer_dominance_flag", "answer_dominance"),
            ("fast_low_quality_flag", "fast_low_quality"),
            ("unfinished_behavior_flag", "unfinished_behavior"),
        ]
        selected = [name for col, name in reason_names if row[col]]
        return ",".join(selected) if selected else "ok"

    workers["suspicious_reasons"] = workers.apply(reasons, axis=1)
    return workers.sort_values(
        ["suspicious_flag", "control_accuracy", "answer_dominance_share", "items_done"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def build_noncontrol_url_metrics(items: pd.DataFrame) -> pd.DataFrame:
    # Для обычных URL смотрим согласованность исполнителей без использования контрольных заданий.
    df = items[
        (items["assignment_status"] == "APPROVED")
        & (items["is_control"] == False)
        & items["url"].notna()
        & items["worker_answer"].notna()
    ].copy()

    grouped = df.groupby(["url", "domain"], dropna=False).agg(
        shown_count=("worker_answer", "count"),
        unique_workers=("worker_id", "nunique"),
        yes_count=("worker_answer", lambda x: (x == "yes").sum()),
        no_count=("worker_answer", lambda x: (x == "no").sum()),
        not_working_count=("worker_answer", lambda x: (x == "not_working").sum()),
    ).reset_index()

    grouped["max_votes"] = grouped[
        ["yes_count", "no_count", "not_working_count"]
    ].max(axis=1)
    # confidence показывает силу большинства, disagreement_score - уровень спорности.
    grouped["confidence"] = grouped.apply(
        lambda row: share(row["max_votes"], row["shown_count"]),
        axis=1,
    )
    grouped["disagreement_score"] = 1 - grouped["confidence"]

    def label(row):
        counts = {
            "yes": row["yes_count"],
            "no": row["no_count"],
            "not_working": row["not_working_count"],
        }
        max_votes = max(counts.values())
        winners = [answer for answer, count in counts.items() if count == max_votes]
        return winners[0] if len(winners) == 1 else "tie"

    grouped["final_label"] = grouped.apply(label, axis=1)
    grouped["is_tie"] = grouped["final_label"] == "tie"
    grouped["enough_votes"] = grouped["shown_count"] >= 3
    # Спорным считаем URL с достаточным числом голосов и слабым большинством или ничьей.
    grouped["is_controversial"] = (
        grouped["enough_votes"]
        & ((grouped["confidence"] <= 0.60) | grouped["is_tie"])
    )
    return grouped.sort_values(
        ["is_controversial", "disagreement_score", "shown_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_project_metrics(
    assignments: pd.DataFrame,
    worker_metrics: pd.DataFrame,
    control_metrics: pd.DataFrame,
    worker_votes: pd.DataFrame,
) -> pd.DataFrame:
    total_assignments = len(assignments)
    status_counts = assignments["assignment_status"].value_counts()
    approved = int(status_counts.get("APPROVED", 0))
    unfinished = int(status_counts.get("EXPIRED", 0) + status_counts.get("SKIPPED", 0))

    votes = worker_votes.copy()
    votes["resolved_is_correct"] = to_bool(votes["resolved_is_correct"])
    votes["is_worker_conflict"] = to_bool(votes["is_worker_conflict"])
    valid_votes = votes[
        (votes["is_worker_conflict"] != True)
        & votes["resolved_is_correct"].notna()
    ]
    # Основная метрика качества проекта: точность валидных голосов на контрольных заданиях.
    control_accuracy = share(
        int((valid_votes["resolved_is_correct"] == True).sum()),
        len(valid_votes),
    )

    eligible_controls = control_metrics[control_metrics["eligible_by_n"] == True]
    problem_controls = control_metrics[control_metrics["flag"] != "ok"]
    suspicious_workers = worker_metrics[worker_metrics["suspicious_flag"]]

    # Метрики собираются в таблицу, чтобы отчет и CSV использовали один источник формул.
    rows = [
        (
            "completion_rate",
            "APPROVED assignments / all assignments",
            share(approved, total_assignments),
            "Операционная стабильность: доля страниц, завершённых и одобренных.",
        ),
        (
            "unfinished_assignment_share",
            "(EXPIRED assignments + SKIPPED assignments) / all assignments",
            share(unfinished, total_assignments),
            "Потери потока: доля EXPIRED + SKIPPED среди всех страниц.",
        ),
        (
            "control_accuracy (основная метрика качества проекта)",
            "correct worker-control votes / valid worker-control votes",
            control_accuracy,
            "Качество разметки: точность исполнителей на контрольных заданиях.",
        ),
        (
            "problem_control_share",
            "(difficult controls + ambiguous controls) / controls with n_valid_votes >= 4",
            share(len(problem_controls), len(eligible_controls)),
            "Качество контрольного пула: доля сложных/неоднозначных контролей среди контролей с n>=4.",
        ),
        (
            "suspicious_worker_share",
            "workers with any risk flag / all workers",
            share(len(suspicious_workers), worker_metrics["worker_id"].nunique()),
            "Риск по исполнителям: доля работников с признаками низкого качества или неустойчивого поведения.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "formula", "value", "why_it_matters"])


def save_control_histogram(control_metrics: pd.DataFrame, out_dir: Path) -> Path:
    # Единственный график оставлен как быстрый визуальный контроль распределения качества контролей.
    eligible = control_metrics[control_metrics["n_valid_votes"] >= 4].copy()
    path = out_dir / "control_correct_share_histogram.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eligible["correct_share"], bins=np.linspace(0, 1, 21), color="#4C78A8")
    ax.axvspan(0.45, 0.55, color="#F58518", alpha=0.18, label="ambiguous")
    ax.axvline(0.40, color="#E45756", linestyle="--", linewidth=1, label="difficult cutoff")
    ax.set_title("Control tasks by correct answer share")
    ax.set_xlabel("Correct share")
    ax.set_ylabel("Control tasks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int) -> str:
    # Markdown-таблица нужна для компактной сводки; полные данные остаются в CSV.
    visible = df.loc[:, columns].head(max_rows).copy()
    for col in visible.columns:
        if pd.api.types.is_float_dtype(visible[col]):
            if col.endswith("_share") or col.endswith("_accuracy") or col in {
                "confidence",
                "correct_share",
                "incorrect_share",
            }:
                visible[col] = visible[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            else:
                visible[col] = visible[col].map(
                    lambda x: f"{int(x)}" if pd.notna(x) and float(x).is_integer()
                    else f"{x:.3f}" if pd.notna(x)
                    else ""
                )

    def clean(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [[clean(value) for value in row] for row in visible.to_numpy()]
    header = [clean(col) for col in visible.columns]
    separator = ["---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    dataset_dir: Path,
    controls_dir: Path,
    assignments: pd.DataFrame,
    items: pd.DataFrame,
    worker_metrics: pd.DataFrame,
    control_metrics: pd.DataFrame,
    problem_controls: pd.DataFrame,
    project_metrics: pd.DataFrame,
    top_suspicious: pd.DataFrame,
    top_controversial: pd.DataFrame,
    chart_path: Path,
):
    status_counts = assignments["assignment_status"].value_counts()
    approved = int(status_counts.get("APPROVED", 0))
    expired = int(status_counts.get("EXPIRED", 0))
    skipped = int(status_counts.get("SKIPPED", 0))
    eligible_controls = control_metrics[control_metrics["eligible_by_n"] == True]

    difficult = int((control_metrics["is_difficult"] == True).sum())
    ambiguous = int((control_metrics["is_ambiguous"] == True).sum())
    control_accuracy = project_metrics.loc[
        project_metrics["metric"] == "control_accuracy (основная метрика качества проекта)", "value"
    ].iloc[0]

    metric_lines = []
    for _, row in project_metrics.iterrows():
        metric_lines.append(
            f"| {row['metric']} | `{row['formula']}` | {format_percent(row['value'])} | {row['why_it_matters']} |"
        )

    problem_columns = [
        "url",
        "gold_answer",
        "n_valid_votes",
        "correct_share",
        "incorrect_share",
        "flag",
    ]
    suspicious_columns = [
        "worker_id",
        "approved_count",
        "items_done",
        "controls_done",
        "control_accuracy",
        "answer_dominance_share",
        "suspicious_reasons",
    ]
    controversial_columns = [
        "url",
        "shown_count",
        "confidence",
        "final_label",
        "yes_count",
        "no_count",
        "not_working_count",
    ]

    summary_metrics = project_metrics.copy()
    summary_metrics["value"] = summary_metrics["value"].map(format_percent)

    # Скрипт генерирует короткую сводку; подробный интерпретационный report.md можно вести отдельно.
    report = f"""# Краткая сводка по Crowd-проекту

## Ключевые значения

- Ассайнменты: {format_number(len(assignments))}
- Item-строки: {format_number(len(items))}
- Исполнители: {format_number(items['worker_id'].nunique())}
- APPROVED: {format_number(approved)}
- EXPIRED: {format_number(expired)}
- SKIPPED: {format_number(skipped)}
- Уникальные контрольные задания: {format_number(len(control_metrics))}
- Контроли с достаточным числом голосов (`n_valid_votes >= 4`): {format_number(len(eligible_controls))}
- Проблемные контроли: {format_number(len(problem_controls))}
- Сложные контроли: {format_number(difficult)}
- Неоднозначные контроли: {format_number(ambiguous)}

## Метрики проекта

{markdown_table(summary_metrics, ["metric", "formula", "value", "why_it_matters"], 10)}

## Топ проблемных контролей

{markdown_table(problem_controls, problem_columns, 10)}

## Топ подозрительных исполнителей

{markdown_table(top_suspicious, suspicious_columns, 10)}

## Топ спорных обычных URL

{markdown_table(top_controversial, controversial_columns, 10)}

## Артефакты

- `{dataset_dir / 'items_long.csv'}` — основной long-датасет.
- `{controls_dir / 'control_task_metrics.csv'}` — все контрольные задания с метриками.
- `{controls_dir / 'problem_controls.csv'}` — difficult/ambiguous контроли.
- `{out_dir / 'project_metrics.csv'}` — ключевые метрики проекта.
- `{out_dir / 'top_suspicious_workers.csv'}` — исполнители с риск-флагами.
- `{out_dir / 'top_controversial_urls.csv'}` — URL с низкой согласованностью голосов.
- `{chart_path}` — график распределения `correct_share` по контролям.
"""

    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Folder with parsed dataset tables")
    parser.add_argument("--controls", required=True, help="Folder with control task metrics")
    parser.add_argument("--out", default="report", help="Output folder")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    controls_dir = Path(args.controls)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_table(find_table(dataset_dir, "items_long"))
    items["is_control"] = to_bool(items["is_control"])
    items["is_correct"] = to_bool(items["is_correct"])

    assignments = read_optional_table(
        dataset_dir / "assignments.parquet",
        dataset_dir / "assignments.csv",
    )
    if assignments is None:
        assignments = derive_assignments(items)

    worker_votes = read_optional_table(
        controls_dir / "worker_control_votes.csv",
        controls_dir / "worker_control_votes.parquet",
    )
    if worker_votes is None:
        raise FileNotFoundError("worker_control_votes.csv is required for report metrics")

    control_metrics = read_table(controls_dir / "control_task_metrics.csv")
    for col in ["eligible_by_n", "is_difficult", "is_ambiguous", "is_conflicted_repeats"]:
        control_metrics[col] = to_bool(control_metrics[col])

    # Собираем минимальный набор артефактов отчета без пересчета исходного датасета.
    worker_metrics = build_worker_metrics(items, assignments, worker_votes)
    noncontrol_urls = build_noncontrol_url_metrics(items)
    project_metrics = build_project_metrics(
        assignments=assignments,
        worker_metrics=worker_metrics,
        control_metrics=control_metrics,
        worker_votes=worker_votes,
    )

    problem_controls = control_metrics[control_metrics["flag"] != "ok"].copy()
    # При сортировке неоднозначных контролей ближе к 0.5 считаем более спорными.
    problem_controls["ambiguity_distance"] = (
        problem_controls["correct_share"] - 0.5
    ).abs()
    problem_controls = problem_controls.sort_values(
        ["is_difficult", "is_ambiguous", "incorrect_share", "ambiguity_distance", "n_valid_votes"],
        ascending=[False, False, False, True, False],
    )
    top_suspicious = worker_metrics[worker_metrics["suspicious_flag"]].head(50).copy()
    top_controversial = noncontrol_urls[noncontrol_urls["is_controversial"]].head(50).copy()

    project_metrics.to_csv(out_dir / "project_metrics.csv", index=False)
    top_suspicious.to_csv(out_dir / "top_suspicious_workers.csv", index=False)
    top_controversial.to_csv(out_dir / "top_controversial_urls.csv", index=False)
    chart_path = save_control_histogram(control_metrics, out_dir)

    write_report(
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        controls_dir=controls_dir,
        assignments=assignments,
        items=items,
        worker_metrics=worker_metrics,
        control_metrics=control_metrics,
        problem_controls=problem_controls,
        project_metrics=project_metrics,
        top_suspicious=top_suspicious,
        top_controversial=top_controversial,
        chart_path=chart_path,
    )

    print("Done")
    print(f"project metrics: {len(project_metrics)}")
    print(f"problem controls: {len(problem_controls)}")
    print(f"top suspicious workers: {len(top_suspicious)}")
    print(f"top controversial URLs: {len(top_controversial)}")
    print(f"report: {(out_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
