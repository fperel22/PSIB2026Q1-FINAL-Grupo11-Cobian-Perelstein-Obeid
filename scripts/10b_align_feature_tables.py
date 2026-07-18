"""
10b_align_feature_tables.py

Construye una cohorte común para comparar características obtenidas con
máscaras manuales y automáticas.

Reglas:
- Las tablas manual y automática deben contener los mismos filenames.
- class y subset deben coincidir para cada filename.
- Una imagen se conserva únicamente si TODAS sus características son finitas
  en ambas fuentes.
- La salida esperada para este proyecto es:
      train = 216
      val   = 48
      test  = 47

Entradas:
    outputs/tables/features_manual.csv
    outputs/tables/features_auto.csv

Salidas:
    outputs/tables_aligned/features_manual.csv
    outputs/tables_aligned/features_auto.csv
    outputs/tables_aligned/excluded_cases.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLUMNS = {
    "filename",
    "class",
    "subset",
    "mask_source",
    "mask_area_px",
    "wavelet_levels",
}

EXPECTED_COUNTS = {
    "train": 216,
    "val": 48,
    "test": 47,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--manual-table",
        type=Path,
        default=Path("outputs/tables/features_manual.csv"),
    )
    parser.add_argument(
        "--auto-table",
        type=Path,
        default=Path("outputs/tables/features_auto.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tables_aligned"),
    )

    return parser.parse_args()


def load_table(path: Path, expected_source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

    df = pd.read_csv(path)

    required = {"filename", "class", "subset", "mask_source"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} no contiene las columnas requeridas: {sorted(missing)}"
        )

    if df["filename"].duplicated().any():
        duplicated = (
            df.loc[df["filename"].duplicated(), "filename"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"{path} contiene filenames duplicados: {duplicated[:10]}"
        )

    invalid_sources = sorted(
        set(df["mask_source"].astype(str)) - {expected_source}
    )
    if invalid_sources:
        raise ValueError(
            f"{path} contiene mask_source inesperados: {invalid_sources}"
        )

    return df


def invalid_feature_names(
    numeric_features: pd.DataFrame,
    row_index: int,
) -> list[str]:
    row = numeric_features.iloc[row_index]

    return [
        column
        for column, value in row.items()
        if not np.isfinite(value)
    ]


def main() -> None:
    args = parse_args()

    manual = load_table(args.manual_table, "manual")
    auto = load_table(args.auto_table, "auto")

    if list(manual.columns) != list(auto.columns):
        raise ValueError(
            "Las tablas manual y automática no tienen exactamente "
            "las mismas columnas y el mismo orden."
        )

    manual_filenames = set(manual["filename"].astype(str))
    auto_filenames = set(auto["filename"].astype(str))

    if manual_filenames != auto_filenames:
        only_manual = sorted(manual_filenames - auto_filenames)
        only_auto = sorted(auto_filenames - manual_filenames)

        raise ValueError(
            "Las tablas no contienen los mismos filenames.\n"
            f"Solo manual: {only_manual[:10]}\n"
            f"Solo automática: {only_auto[:10]}"
        )

    # Reordenar la tabla automática según el orden de la tabla manual.
    auto_by_filename = auto.set_index("filename", drop=False)
    auto = (
        auto_by_filename
        .loc[manual["filename"].astype(str)]
        .reset_index(drop=True)
    )
    manual = manual.reset_index(drop=True)

    if not manual["class"].equals(auto["class"]):
        raise ValueError(
            "Las clases manual/automática no coinciden por filename."
        )

    if not manual["subset"].equals(auto["subset"]):
        raise ValueError(
            "Los subsets manual/automático no coinciden por filename."
        )

    feature_columns = [
        column
        for column in manual.columns
        if column not in META_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No se encontraron columnas de características.")

    manual_numeric = manual[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    auto_numeric = auto[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    manual_valid = np.isfinite(
        manual_numeric.to_numpy(dtype=float)
    ).all(axis=1)

    auto_valid = np.isfinite(
        auto_numeric.to_numpy(dtype=float)
    ).all(axis=1)

    common_valid = manual_valid & auto_valid

    excluded_rows: list[dict[str, object]] = []

    for index in np.flatnonzero(~common_valid):
        manual_invalid = invalid_feature_names(
            manual_numeric,
            int(index),
        )
        auto_invalid = invalid_feature_names(
            auto_numeric,
            int(index),
        )

        reasons: list[str] = []

        if manual_invalid:
            reasons.append(
                "manual:features_no_finitas="
                + ";".join(manual_invalid)
            )

        if auto_invalid:
            reasons.append(
                "auto:features_no_finitas="
                + ";".join(auto_invalid)
            )

        excluded_rows.append(
            {
                "filename": manual.loc[index, "filename"],
                "class": manual.loc[index, "class"],
                "subset": manual.loc[index, "subset"],
                "manual_mask_area_px": manual.loc[
                    index,
                    "mask_area_px",
                ],
                "auto_mask_area_px": auto.loc[
                    index,
                    "mask_area_px",
                ],
                "manual_wavelet_levels": manual.loc[
                    index,
                    "wavelet_levels",
                ],
                "auto_wavelet_levels": auto.loc[
                    index,
                    "wavelet_levels",
                ],
                "manual_valid": bool(manual_valid[index]),
                "auto_valid": bool(auto_valid[index]),
                "reason": " | ".join(reasons),
            }
        )

    manual_aligned = manual.loc[common_valid].reset_index(drop=True)
    auto_aligned = auto.loc[common_valid].reset_index(drop=True)

    if not manual_aligned[
        ["filename", "class", "subset"]
    ].equals(
        auto_aligned[["filename", "class", "subset"]]
    ):
        raise RuntimeError(
            "La alineación final manual/automática no coincide."
        )

    obtained_counts = {
        subset: int(
            (manual_aligned["subset"] == subset).sum()
        )
        for subset in ("train", "val", "test")
    }

    if obtained_counts != EXPECTED_COUNTS:
        raise ValueError(
            "Los conteos de la cohorte común no coinciden con los "
            f"esperados. Obtenido={obtained_counts}, "
            f"esperado={EXPECTED_COUNTS}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manual_path = args.output_dir / "features_manual.csv"
    auto_path = args.output_dir / "features_auto.csv"
    excluded_path = args.output_dir / "excluded_cases.csv"

    manual_aligned.to_csv(manual_path, index=False)
    auto_aligned.to_csv(auto_path, index=False)

    excluded = pd.DataFrame(excluded_rows)
    excluded.to_csv(excluded_path, index=False)

    print("Alineación terminada.")
    print(f"Filas originales: {len(manual)}")
    print(f"Filas conservadas: {len(manual_aligned)}")
    print(f"Filas excluidas: {len(excluded)}")
    print(f"Conteos finales: {obtained_counts}")
    print(f"Guardado: {manual_path}")
    print(f"Guardado: {auto_path}")
    print(f"Guardado: {excluded_path}")


if __name__ == "__main__":
    main()