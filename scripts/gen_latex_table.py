import argparse
from collections import defaultdict

import pandas as pd
import wandb


def main(args):

    api = wandb.Api()

    # api.default_entity by default
    entity = api.default_entity if args.entity is None else args.entity

    # get runs from the project
    runs = api.runs(f"{entity}/{args.project}")
    print(f"Find {len(runs)} runs in {entity}/{args.project}")

    # prepare table content
    results = defaultdict(list)
    for run in runs:
        if "test/acc" not in run.summary:
            continue
        model_name = run.config["model/_target_"].split(".")[-1]
        results["Model"].append(model_name)
        results["Accuracy"].append(run.summary["test/acc"])

    df = pd.DataFrame(results)

    # generating the table by using: Pandas DataFrame.style.to_latex
    # https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
    highlight_metrics = ["Accuracy"]
    style = df.style.highlight_max(
        axis=0, subset=highlight_metrics, props="textbf:--rwrap;"
    )
    style = style.hide(axis="index")

    print()
    print(r"\usepackage{booktabs}")
    print()
    print(
        style.to_latex(
            caption=args.caption,
            hrules=True,
            position=args.position,
            position_float="centering",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default="mnist")
    parser.add_argument("--caption", default="Model accuracy on MNIST")
    parser.add_argument("--position", default="ht")
    args = parser.parse_args()
    main(args)
