from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()

def head(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    n: int = typer.Option(5, "--n", help="Сколько строк вывести."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
):
    """
    Вывести первые n строк датасета.
    Полезно для быстрых проверок структуры данных.
    """
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    typer.echo(df.head(n).to_string(index=False))

def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(10, help="Сколько top значений показывать в категориальных признаках."),
    title: str = typer.Option(None, help="Заголовок отчёта (по умолчанию имя файла)."),
    min_missing_share: float = typer.Option(0.3, help="Порог проблемной доли пропусков."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)

    # обновленный вызов: используем top_k_categories
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество данных с учётом новых эвристик
    quality_flags = compute_quality_flags(summary, missing_df, df)
    if not missing_df.empty:
        problematic = missing_df[missing_df["missing_share"] >= min_missing_share]
        problematic_cols = problematic.index.tolist()
    else:
        problematic_cols = []

    quality_flags["problematic_columns_by_missing"] = problematic_cols
    quality_flags["n_problematic_columns"] = len(problematic_cols)


    # Добавим порог проблемных пропусков в флаги
    quality_flags["problematic_columns_by_missing"] = (
        missing_df["missing_share"] >= min_missing_share
    ).sum()


    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)

    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)

    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)

    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    report_title = title or f"EDA-отчёт для {Path(path).name}"

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {report_title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Параметры отчёта\n\n")
        f.write(f"- max_hist_columns = **{max_hist_columns}**\n")
        f.write(f"- top_k_categories = **{top_k_categories}**\n")
        f.write(f"- min_missing_share = **{min_missing_share:.0%}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Колонки с пропусками выше порога ({min_missing_share:.0%}):\n")
        if quality_flags["n_problematic_columns"] == 0:
            f.write("  - нет\n")
        else:
            for col in quality_flags["problematic_columns_by_missing"]:
                f.write(f"  - {col}\n")
        f.write("\n")

        # Новые эвристики
        f.write(f"- Константные колонки: **{quality_flags['has_constant_columns']}**\n")
        f.write(f"- Высокая кардинальность категорий: **{quality_flags['has_high_cardinality_categoricals']}**\n")
        f.write(f"- Дубликаты идентификаторов: **{quality_flags['has_suspicious_id_duplicates']}**\n")
        f.write(f"- Слишком много нулей: **{quality_flags['has_many_zero_values']}**\n")
        f.write(f"- Колонок с пропусками >= порога: **{quality_flags['problematic_columns_by_missing']}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные колонки\n\n")
        if not top_cats:
            f.write("Категориальные признаки отсутствуют.\n\n")
        else:
            f.write("См. файлы в `top_categories/`.\n\n")

        f.write("## Гистограммы\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Графики
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован: {md_path}")


if __name__ == "__main__":
    app()
