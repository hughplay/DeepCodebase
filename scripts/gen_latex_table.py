import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import wandb


def main(args):

    api = wandb.Api()

    entity = api.default_entity if args.entity is None else args.entity

    def filter_runs(filters=None, sort=None):
        runs = api.runs(f"{entity}/{args.project}", filters=filters)
        runs = [run for run in runs if ("test/acc" in run.summary)]
        if sort is not None:
            runs = sorted(runs, key=sort)
        print(f"Find {len(runs)} runs in {entity}/{args.project}")
        return runs

    latex_str = getattr(sys.modules[__name__], f"{args.table}_table")(
        filter_runs
    )
    print(latex_str)


def gen_latex(
    style, caption, position="ht", small=True, save_path=None, **kwargs
):
    print(r"\usepackage{booktabs}")
    print()
    latex_str = style.to_latex(
        caption=caption,
        hrules=True,
        position=position,
        position_float="centering",
        **kwargs,
    )
    if small:
        latex_str = latex_str.replace("\\centering", "\\centering\n\\small")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            f.write(latex_str)
    return latex_str


def baseline_table(filter_runs: callable):
    # filters = {"tags": {"$in": [baseline]}}
    filters = {}
    runs = filter_runs(filters, sort=lambda run: run.summary["test/acc"])
    results = defaultdict(list)
    for run in runs:
        if "test/acc" not in run.summary:
            continue
        model_name = run.config["model/_target_"].split(".")[-1]
        results["Model"].append(model_name)
        results["Accuracy"].append(run.summary["test/acc"])

    df = pd.DataFrame(results)

    highlight_metrics = ["Accuracy"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.hide(axis="index")

    str_latex = gen_latex(
        style,
        "Performance on the MNIST dataset. ",
        save_path="docs/tables/baseline.tex",
        label="tab:baseline",
        column_format="lr",
        position="t",
    )

    return str_latex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--table", default="main")
    parser.add_argument("--project", default="mnist")
    parser.add_argument("--caption", default="Model performance.")
    args = parser.parse_args()
    main(args)
