"""A camera model chooser and an exportable table from accepted ablation metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

from demos.v3_models import CameraModel, load_catalog

PROFILE_NAMES = {"G0": "GRU mean/max", "G-mean": "GRU mean", "G-attn": "GRU attention",
                 "G-bi": "BiGRU", "L0": "LSTM", "T0": "Transformer"}


def ranked_models(catalog) -> list[CameraModel]:
    seen = set()
    rows = []
    for model in catalog.values():
        if model.run_id not in seen:
            rows.append(model)
            seen.add(model.run_id)
    return rows


def write_table(catalog, output: Path) -> None:
    rows = ranked_models(catalog)
    lines = ["# Bảng ablation V3 để chọn model camera", "",
             "Nguồn: inventory full42 đã nghiệm thu, freeze `b4fd56f03a57`. "
             "Tất cả chỉ số bảng là validation mean qua seeds 42/43/44; ± là population std, đơn vị điểm phần trăm.", "",
             "Ưu tiên `T0__HF` nếu nhìn rõ tay và mặt: đứng đầu validation. "
             "Thử `T0__H` khi muốn nhận diện chỉ từ tay, và `G-bi__HBF` để đối chiếu với BiGRU tốt nhất. "
             "Model B/F/BF ít chính xác hơn rõ rệt trong campaign này.", "",
             "H = tay, B = thân trên, F = mặt. Các vùng không được chọn không tham gia fusion. "
             "Auto detect vẫn dùng tay để xác định đầu/cuối động tác.", "",
             "## Config tốt nhất của từng kiến trúc", "",
             "| Kiến trúc | Config | Macro-F1 (%) | Top-1 (%) | Top-3 (%) | Parameters |",
             "|---|---|---:|---:|---:|---:|"]
    for profile, name in PROFILE_NAMES.items():
        row = next(model for model in rows if model.run_id.split("__")[0] == profile)
        lines.append(f"| {name} | `{row.run_id}` | {row.macro_f1 * 100:.2f} ± {row.macro_f1_std * 100:.2f} "
                     f"| {row.top1 * 100:.2f} | {row.top3 * 100:.2f} | {row.parameters:,} |")
    lines += ["", "## Toàn bộ 42 configs", "",
              "| Hạng | Config | Vùng | Macro-F1 (%) | Top-1 (%) | Top-3 (%) | Parameters |",
              "|---:|---|---|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, 1):
        lines.append(f"| {rank} | `{row.run_id}` | {' + '.join(row.regions)} "
                     f"| {row.macro_f1 * 100:.2f} ± {row.macro_f1_std * 100:.2f} "
                     f"| {row.top1 * 100:.2f} | {row.top3 * 100:.2f} | {row.parameters:,} |")
    lines += ["", "## Cách đọc", "",
              "Ranking dựa trên validation theo rule đã freeze: mean Macro-F1, std thấp hơn, "
              "parameters ít hơn, run ID. Số parameters giúp hiểu kích thước model; bảng chưa đo latency camera.", "",
              "Winner `T0__HF` đã được test riêng: Macro-F1 87,30% ± 0,77, Top-1 88,20% ± 1,07, "
              "Top-3 98,34% ± 0,52 qua ba seeds. Các configs khác chưa được so sánh trên test cohort.", "",
              "Điểm softmax trên camera và validation accuracy là hai phép đo khác nhau. "
              "Hãy ghi một clip rồi chạy lại cùng clip ở các model để so sánh hành vi thực tế."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chooser(parent, catalog, on_choose):
    import tkinter as tk
    from tkinter import ttk

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(2, weight=1)
    title = ttk.Frame(parent, padding=(18, 14))
    title.grid(row=0, column=0, sticky="ew")
    ttk.Label(title, text="Chọn model từ ablation V3", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    ttk.Label(title, text="Validation mean ± std qua 3 seeds · 42 configs · H: tay, B: thân trên, F: mặt").pack(anchor="w", pady=6)
    ttk.Label(title, text="Gợi ý: T0__HF cho tay + mặt · T0__H cho tay · G-bi__HBF để thử BiGRU").pack(anchor="w")
    filters = ttk.Frame(parent, padding=(18, 4))
    filters.grid(row=1, column=0, sticky="ew")
    profile_var = tk.StringVar(value="Tất cả")
    region_var = tk.StringVar(value="Tất cả")
    ttk.Label(filters, text="Kiến trúc").pack(side="left", padx=(0, 6))
    profiles = ttk.Combobox(filters, textvariable=profile_var, state="readonly", width=14,
                           values=["Tất cả", *PROFILE_NAMES])
    profiles.pack(side="left", padx=(0, 16))
    ttk.Label(filters, text="Vùng").pack(side="left", padx=(0, 6))
    regions = ttk.Combobox(filters, textvariable=region_var, state="readonly", width=8,
                          values=["Tất cả", "H", "B", "F", "HB", "HF", "BF", "HBF"])
    regions.pack(side="left")
    body = ttk.Frame(parent, padding=(18, 8))
    body.grid(row=2, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    columns = ("rank", "model", "region", "f1", "top1", "top3", "params")
    table = ttk.Treeview(body, columns=columns, show="headings", selectmode="browse")
    labels = ("Hạng", "Model", "Vùng", "Macro-F1 (%)", "Top-1 (%)", "Top-3 (%)", "Parameters")
    widths = (55, 145, 70, 175, 100, 100, 120)
    for key, label, width in zip(columns, labels, widths):
        table.heading(key, text=label)
        table.column(key, width=width, minwidth=width, anchor="w" if key == "model" else "center")
    table.grid(row=0, column=0, sticky="nsew")
    scroll = ttk.Scrollbar(body, orient="vertical", command=table.yview)
    scroll.grid(row=0, column=1, sticky="ns")
    horizontal = ttk.Scrollbar(body, orient="horizontal", command=table.xview)
    horizontal.grid(row=1, column=0, sticky="ew")
    table.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
    rows = ranked_models(catalog)

    def refresh(event=None):
        table.delete(*table.get_children())
        for rank, row in enumerate(rows, 1):
            profile, feature = row.run_id.split("__")
            if profile_var.get() not in ("Tất cả", profile) or region_var.get() not in ("Tất cả", feature):
                continue
            table.insert("", "end", iid=row.run_id, values=(rank, row.run_id, feature,
                f"{row.macro_f1 * 100:.2f} ± {row.macro_f1_std * 100:.2f}",
                f"{row.top1 * 100:.2f}", f"{row.top3 * 100:.2f}", f"{row.parameters:,}"))
        if table.get_children():
            table.selection_set(table.get_children()[0])

    def choose(event=None):
        selection = table.selection()
        if selection:
            on_choose(selection[0])

    profiles.bind("<<ComboboxSelected>>", refresh)
    regions.bind("<<ComboboxSelected>>", refresh)
    table.bind("<Double-1>", choose)
    table.bind("<Return>", choose)
    footer = ttk.Frame(parent, padding=(18, 12))
    footer.grid(row=3, column=0, sticky="ew")
    ttk.Button(footer, text="Dùng model được chọn", command=choose).pack(side="left", padx=(0, 14))
    ttk.Label(footer, text="Chọn một dòng rồi bấm Dùng model · Giữ seed đang chọn trên tab Camera").pack(side="left")
    refresh()
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ABLATION_STUDY.md"))
    args = parser.parse_args()
    write_table(load_catalog(), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
