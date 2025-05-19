# bias_evaluator.py
import itertools
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


class BiasEvaluator:
    def __init__(
        self,
        passion_exp: str = "experiment_standard_split_conditions",
        eval_data_path="../../assets/evaluation",
        dataset_dir: Union[str, Path] = "../../data/PASSION/",
        meta_data_file: Union[str, Path] = "label.csv",
        split_file: Union[str, Path, None] = "PASSION_split.csv",
        threshold_eq_rates: float = 0.015,
        target_names: [str] = None,
        labels: [str] = None,
    ):
        self.passion_exp = passion_exp
        self.eval_data_path = Path(eval_data_path)
        self.dataset_dir = Path(dataset_dir)
        self.out_dir = self.eval_data_path / passion_exp
        os.makedirs(self.out_dir, exist_ok=True)

        self.threshold_eq_rates = threshold_eq_rates

        self.target_names = target_names
        self.labels = labels

        # todo: only read once in the whole pipeline
        self.df_labels = pd.read_csv(self.dataset_dir / meta_data_file)
        self.df_split = pd.read_csv(self.dataset_dir / split_file)

    def get_subgroup_metric_results_file_name(self, add_run_info):
        return (
            self.out_dir
            / f"subgroup_metric_results_{self.passion_exp}__{add_run_info}.csv"
        )

    def get_predictions_with_meta_data_file_name(self, add_run_info):
        return (
            self.out_dir
            # todo cleanup
            / f"predictions_with_metadata_{self.passion_exp}__{add_run_info}_img_lvl.csv"
            # / "analysis_experiment_standard_split_conditions_passion__big_model_test_img_lvl.csv"
            # / "predictions_with_metadata_experiment_standard_split_conditions_passion__imagenet_tiny__Test_img_lvl.csv"
        )

    @staticmethod
    def _parse_image_paths(s: str):
        return (
            s.replace("\n", " ")
            .replace("'", '"')
            .replace("[", "")
            .replace("]", "")
            .split()
        )

    @staticmethod
    def _parse_numpy_array(s: str):
        return np.fromstring(s.strip("[]").replace("\n", " "), sep=" ", dtype=int)

    @staticmethod
    def _extract_subject_id(path: str):
        match = re.search(r"([A-Za-z]+[0-9]+)", path)
        return str(match.group(1)).strip() if match else np.nan

    def _get_data_with_metadata_from_csv(
        self, create_data: bool = True, add_run_info: str = ""
    ):
        predictions_n_meta_data_file = self.get_predictions_with_meta_data_file_name(
            add_run_info
        )
        if create_data:
            input_file = self.eval_data_path / f"{self.passion_exp}__{add_run_info}.csv"
            df_results = pd.read_csv(input_file)
            df_results = df_results.iloc[[-1]]

            df_results["FileNames"] = df_results["FileNames"].apply(
                self._parse_image_paths
            )
            df_results["Indices"] = df_results["Indices"].apply(self._parse_numpy_array)
            df_results["EvalTargets"] = df_results["EvalTargets"].apply(
                self._parse_numpy_array
            )
            df_results["EvalPredictions"] = df_results["EvalPredictions"].apply(
                self._parse_numpy_array
            )

            data = self._aggregate_data_with_metadata(df_results)
            data.to_csv(predictions_n_meta_data_file, index=False)
            return data

        return pd.read_csv(predictions_n_meta_data_file)

    def _get_data_with_metadata(
        self, df_results: pd.DataFrame = None, add_run_info: str = ""
    ):
        predictions_n_meta_data_file = self.get_predictions_with_meta_data_file_name(
            add_run_info
        )
        data = self._aggregate_data_with_metadata(df_results)
        data.to_csv(predictions_n_meta_data_file, index=False)
        return data

    def _aggregate_data_with_metadata(self, df_results):
        output_rows = []
        seen_subject_ids = set()
        for _, row in df_results.iterrows():
            for img_name, idx, lbl, pred in zip(
                row["FileNames"],
                row["Indices"],
                row["EvalTargets"],
                row["EvalPredictions"],
            ):
                subject_id = self._extract_subject_id(img_name)
                labels = self.df_labels[
                    self.df_labels["subject_id"] == subject_id
                ].iloc[0]
                split = self.df_split[self.df_split["subject_id"] == subject_id].iloc[0]
                output_rows.append(
                    {
                        "correct": lbl == pred,
                        "image_path": img_name,
                        "index": idx,
                        "targets": lbl,
                        "predictions": pred,
                        **labels.to_dict(),
                        **split.drop("subject_id").to_dict(),
                    }
                )
                seen_subject_ids.add(subject_id)
        print(f"{len(seen_subject_ids)} unique subjects processed.")
        data = pd.DataFrame(output_rows)
        return data

    def _print_classification_stats(self, y_true, y_pred, balanced_accuracy):
        print(
            classification_report(
                y_true,
                y_pred,
                labels=self.labels,
                target_names=self.target_names,
                zero_division=0,
            )
        )
        print(f"Balanced Accuracy: {balanced_accuracy}")

    def _get_confusion_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        print("Confusion Matrix:")
        print(cm)
        print(f"y_true: {set(y_true)}")
        print(f"y_pred: {set(y_pred)}")

        cumulated = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "tpr": [], "fpr": []}
        per_class = {}

        for i, name in enumerate(self.target_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            tpr = round((tp / (tp + fn)), 2) if (tp + fn) else 0
            fpr = round((fp / (fp + tn)), 2) if (fp + tn) else 0

            cumulated["tp"] += tp
            cumulated["fp"] += fp
            cumulated["fn"] += fn
            cumulated["tn"] += tn
            cumulated["tpr"].append(tpr)
            cumulated["fpr"].append(fpr)

            per_class[name] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "TPR": tpr,
                "FPR": fpr,
            }
            print(
                f"{name} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, TPR: {tpr}, FPR: {fpr}"
            )

        tp_fn = cumulated["tp"] + cumulated["fn"]
        micro_tpr = round(cumulated["tp"] / tp_fn, 2) if tp_fn else 0
        fp_tn = cumulated["fp"] + cumulated["tn"]
        micro_fpr = round(cumulated["fp"] / fp_tn, 2) if fp_tn else 0
        macro_tpr = round(np.mean(cumulated["tpr"]), 2)
        macro_fpr = round(np.mean(cumulated["fpr"]), 2)

        print(
            f"cumulated - TP: {cumulated['tp']}, FP: {cumulated['fp']}, FN: {cumulated['fn']}, TN: {cumulated['tn']}"
            f", macro-TPR (avg): {macro_tpr}, macro-FPR (avg): {macro_fpr}"
            f", micro-TPR: {micro_tpr}, micro-FPR: {micro_fpr}"
        )

        return {
            "macro-tpr": macro_tpr,
            "macro-fpr": macro_fpr,
            "micro-tpr": micro_tpr,
            "micro-fpr": micro_fpr,
            "per_class": per_class,
            "cumulated": {
                "TP": cumulated["tp"],
                "FP": cumulated["fp"],
                "FN": cumulated["fn"],
                "TN": cumulated["tn"],
            },
            "balancedAcc": balanced_accuracy_score(y_true, y_pred),
        }

    # todo potentially add aif360_results
    def run_full_evaluation(
        self,
        analysis_name: str,
        data: pd.DataFrame = None,
        add_run_info: str = None,
        run_detailed_evaluation: bool = False,
    ):
        if data is None:
            df = self._get_data_with_metadata_from_csv(
                create_data=True, add_run_info=add_run_info
            )
        else:
            df = self._get_data_with_metadata(
                df_results=data, add_run_info=add_run_info
            )

        print(
            f"********************* Overall Evaluation - {analysis_name} *********************"
        )
        y_true = df["targets"]
        y_pred = df["predictions"]
        self._calculate_metrics(y_pred, y_true)

        if run_detailed_evaluation:
            print(
                f"********************* Detailed Evaluation - {analysis_name} *********************"
            )
            df["ageGroup"] = self._generate_age_group(df)
            grouping_columns = ["fitzpatrick", "sex", "ageGroup", "country"]
            self._detailed_evaluation(df, grouping_columns, add_run_info)

    def _generate_age_group(self, df):
        bins = range(0, 101, 5)
        return pd.cut(
            df["age"],
            bins=bins,
            labels=[f"{i:02}-{i + 4:02}" for i in bins[:-1]],
            right=False,
        )

    def collect_subgroup_results(self, data, group_by: list[str]):
        def to_pascal_case(s: str) -> str:
            return "".join(word.capitalize() for word in s.split("_"))

        grouped = sorted(data.groupby(group_by, observed=False))
        results = []
        result_keys = None

        for group_values, _df in grouped:
            if isinstance(group_values, str):
                group_values = (group_values,)  # Ensure tuple

            support = _df.shape[0]
            if support == 0:
                print(f"no support: group_by: {group_by}, group_values: {group_values}")
                results.append(
                    {
                        **{attr: val for attr, val in zip(group_by, group_values)},
                        "Support": 0,
                    }
                )
                continue

            group_info = ", ".join(
                f"{to_pascal_case(attr)}: {val}"
                for attr, val in zip(group_by, group_values)
            )
            print("~" * 20 + f" {group_info}, Support: {support} " + "~" * 20)

            y_true = data["targets"][_df.index.values]
            y_pred = data["predictions"][_df.index.values]

            metrics = self._calculate_metrics(y_pred, y_true)

            result = {
                **{attr: val for attr, val in zip(group_by, group_values)},
                "Support": support,
                "Macro-TPR": metrics["macro-tpr"],
                "Macro-FPR": metrics["macro-fpr"],
                "Micro-TPR": metrics["micro-tpr"],
                "Micro-FPR": metrics["micro-fpr"],
                "TP": metrics["cumulated"]["TP"],
                "FP": metrics["cumulated"]["FP"],
                "FN": metrics["cumulated"]["FN"],
                "TN": metrics["cumulated"]["TN"],
                "balancedAcc": metrics["balancedAcc"],
            }
            if not result_keys:
                result_keys = list(result.keys())

            # Add per-class metrics
            for class_name, values in metrics["per_class"].items():
                for metric_name, metric_val in values.items():
                    result[f"{class_name}_{metric_name}"] = metric_val

            results.append(result)

        return pd.DataFrame(results), result_keys

    def _calculate_metrics(self, y_pred, y_true):
        metrics = self._get_confusion_metrics(y_true, y_pred)
        self._print_classification_stats(y_true, y_pred, metrics["balancedAcc"])
        return metrics

    def _detailed_evaluation(self, data, grouping_columns, add_run_info):
        dfs = []
        group_by_key = "GroupBy"
        report = {}
        result_keys = None
        for group in self._get_all_group_combinations(grouping_columns):
            subgroup_df, result_keys = self.collect_subgroup_results(
                data, group_by=group
            )
            subgroup_df[group_by_key] = ", ".join(group)
            dfs.append(subgroup_df)

            macro_tpr_avg = subgroup_df["Macro-TPR"].mean()
            macro_fpr_avg = subgroup_df["Macro-FPR"].mean()

            privileged = []
            underprivileged = []
            avgprivileged = []
            unclear = []
            no_support = []

            for _, row in subgroup_df.iterrows():
                support = row["Support"]
                label = (
                    ", ".join(str(row[col]) for col in group)
                    + f"; Support: {support:>4}"
                )
                if support > 0:
                    macro_tpr = row["Macro-TPR"]
                    macro_fpr = row["Macro-FPR"]

                    # Compare TPR
                    tpr_privilege = 0
                    tpr_diff = macro_tpr - macro_tpr_avg
                    if abs(tpr_diff) <= self.threshold_eq_rates:
                        tpr_comp = f"TPR ~ ({macro_tpr:.2f})"
                    elif tpr_diff > 0:
                        tpr_comp = f"TPR ↑ ({macro_tpr:.2f})"
                        tpr_privilege = 1
                    else:
                        tpr_comp = f"TPR ↓ ({macro_tpr:.2f})"
                        tpr_privilege = -1

                    # Compare FPR
                    fpr_privilege = 0
                    fpr_diff = macro_fpr - macro_fpr_avg
                    if abs(fpr_diff) <= self.threshold_eq_rates:
                        fpr_comp = f"FPR ~ ({macro_fpr:.2f})"
                    elif fpr_diff > 0:
                        fpr_comp = f"FPR ↑ ({macro_fpr:.2f})"
                        fpr_privilege = -1
                    else:
                        fpr_comp = f"FPR ↓ ({macro_fpr:.2f})"
                        fpr_privilege = 1

                    reason = f"{tpr_comp}, {fpr_comp}"
                    info = (label, reason)

                    privilege = tpr_privilege + fpr_privilege
                    if privilege > 0:
                        privileged.append(info)
                    elif privilege < 0:
                        underprivileged.append(info)
                    # privilege = 0
                    elif tpr_privilege == fpr_privilege:  # both are 0
                        avgprivileged.append(info)
                    else:
                        unclear.append(info)  # one is +1, one -1
                else:
                    no_support.append((label, None))

            report_key = ", ".join(group)
            report[report_key] = {
                "macro_tpr_avg": macro_tpr_avg,
                "macro_fpr_avg": macro_fpr_avg,
                "categories": {
                    "privileged": privileged,
                    "underprivileged": underprivileged,
                    "average": avgprivileged,
                    "unclear": unclear,
                    "no support": no_support,
                },
            }
        final_df = pd.concat(dfs, ignore_index=True)
        all_other_cols = list(set(final_df.columns) - set(result_keys) - {group_by_key})
        all_other_cols.sort()
        ordered_cols = [group_by_key, *result_keys, *all_other_cols]
        float_cols = final_df.select_dtypes(include="float").columns
        final_df[float_cols] = final_df[float_cols].round(3)

        final_df.to_csv(
            self.get_subgroup_metric_results_file_name(add_run_info),
            columns=ordered_cols,
            index=False,
        )
        # Print privilege report
        for group_key, group_report in report.items():
            print(f"\n=== Grouping: {group_key} ===")
            print(
                f"macro-TPR avg: {group_report['macro_tpr_avg']}; macro-FPR avg: {group_report['macro_fpr_avg']}"
            )
            for category, entries in group_report["categories"].items():
                print(f"{category}:")
                for label, reasons in entries:
                    if reasons:
                        print(f"  {label} - Reasons: {reasons}")
                    else:
                        print(f"  {label}")
                if len(entries) == 0:
                    print("  -")

    def _get_all_group_combinations(self, grouping_columns):
        group_combinations = []
        for r in range(1, len(grouping_columns) + 1):
            group_combinations.extend(
                [list(comb) for comb in itertools.combinations(grouping_columns, r)]
            )
        return group_combinations

    def _create_split(
        self,
        original_df,
        stratify_cols: [str] = None,
        print_unknown_stratification_issues: bool = False,
        seed: int = 42,
    ):
        seed = 32
        df = original_df.copy()

        # Filter the TRAIN data
        df_save_single_records = pd.DataFrame()
        if stratify_cols is None:
            stratify_str = "none" + f"__seed_{seed}"
        else:
            stratify_str = "_".join(stratify_cols) + f"__seed_{seed}"
            print(f"filtering stratification potential issues for: {stratify_str}")
            lonely_subject_ids = []
            if "fitzpatrick" in stratify_cols:
                lonely_subject_ids.extend(["AA00970059", "AA00971417", "AA00971384"])
            if "country" in stratify_cols:
                lonely_subject_ids.extend(["AA00970059"])
            if "country" in stratify_cols and "fitzpatrick" in stratify_cols:
                lonely_subject_ids.extend(
                    [
                        "AA00970095",
                        "AA00970651",
                        "AA00970828",
                        "AA00971108",
                        "AA00971141",
                        "AA00971251",
                        "AA00971417",
                        "AA00971972",
                    ]
                )
                if "sex" in stratify_cols:
                    lonely_subject_ids.extend(
                        [
                            "AA00970134",
                            "AA00970309",
                            "AA00970661",
                            "AA00970743",
                            "AA00970872",
                            "AA00971265",
                            "AA00971325",
                            "AA00971347",
                            "AA00971351",
                            "AA00971833",
                            "AA00972003",
                        ]
                    )
            if len(lonely_subject_ids) > 0:
                lonely_subject_mask = df["subject_id"].isin(lonely_subject_ids)
                df_save_single_records = df[lonely_subject_mask]
                df = df[~lonely_subject_mask]
            print(f"df_save_single_records: {df_save_single_records}")

        print(f"stratification ongoing with: {stratify_str}")

        train_df = df[df["Split"] == "TRAIN"].copy()
        test_df = df[df["Split"] == "TEST"].copy()

        if print_unknown_stratification_issues and stratify_cols:
            group_sizes = train_df.groupby(stratify_cols).size()
            rare_combinations = group_sizes[group_sizes < 2].reset_index()
            rare_rows = train_df.merge(rare_combinations, on=stratify_cols, how="inner")
            print("🔍 Records with too few samples for stratification:")
            print(rare_rows)

        stratify_vals = (
            train_df[stratify_cols].astype(str).agg("_".join, axis=1)
            if stratify_cols is not None
            else None
        )
        label_col = ["conditions_PASSION", "impetig"]
        X = train_df.drop(columns=label_col + ["Split"])
        y = train_df[label_col]

        # Perform the split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=stratify_vals, random_state=seed
        )

        # Label splits
        train_split = X_train[["subject_id"]].copy()
        train_split["Split"] = "TRAIN"

        val_split = X_val[["subject_id"]].copy()
        if not df_save_single_records.empty:
            val_split = pd.concat([val_split, df_save_single_records[["subject_id"]]])
        val_split["Split"] = "VALIDATION"

        test_split = test_df[["subject_id", "Split"]].copy()

        # Combine all
        final_df = pd.concat([train_split, val_split, test_split], axis=0).reset_index(
            drop=True
        )
        final_df.to_csv(f"split_dataset__{stratify_str}.csv", index=False)
        return final_df, stratify_str

    def run_split_distribution_evaluation(
        self, test_split: bool = False, create_splits: bool = False
    ):
        interesting_cols = [
            "country",
            "sex",
            "fitzpatrick",
            "impetig",
            "conditions_PASSION",
            "ageGroup",
        ]
        # todo: this is on the subject only the split, it could be all subjects with multiple pictures in one split

        splits_to_evaluate = [(self.df_split, "PASSION_split")]
        if create_splits:
            df = self.df_labels.copy().merge(self.df_split, on="subject_id", how="left")
            df.reset_index(drop=True, inplace=True)
            splits_to_evaluate.append((self._create_split(df)))
            splits_to_evaluate.append(
                (self._create_split(df, ["conditions_PASSION", "impetig"]))
            )
            splits_to_evaluate.append(
                (self._create_split(df, ["conditions_PASSION", "impetig", "country"]))
            )
            splits_to_evaluate.append(
                (
                    self._create_split(
                        df, ["conditions_PASSION", "impetig", "fitzpatrick"]
                    )
                )
            )
            splits_to_evaluate.append(
                (
                    self._create_split(
                        df, ["conditions_PASSION", "impetig", "country", "fitzpatrick"]
                    )
                )
            )
            splits_to_evaluate.append(
                (
                    self._create_split(
                        df,
                        [
                            "conditions_PASSION",
                            "impetig",
                            "country",
                            "fitzpatrick",
                            "sex",
                        ],
                    )
                )
            )

        for split, split_name in splits_to_evaluate:
            print(f"Split analysis of {split_name}")
            # if not test_split:
            df = self.df_labels.copy().merge(split, on="subject_id", how="left")
            df.reset_index(drop=True, inplace=True)
            # else:
            #     df = self.df_labels
            df["ageGroup"] = self._generate_age_group(df)

            # if test_split:
            #     passion_labels = ["impetig", "conditions_PASSION"]
            #     # todo: fix
            #     df = df[df["fitzpatrick"] != 1]
            #
            #     X = df.drop(columns=passion_labels)  # all columns except target
            #     y = df[passion_labels]
            #     # stratify_cols = y
            #     interesting_cols_stratification = ["sex", "country", "fitzpatrick", "impetig", "conditions_PASSION"]
            #     # interesting_cols_stratification = ["fitzpatrick", "impetig"]
            #     # interesting_cols_stratification = ["fitzpatrick", "conditions_PASSION"]
            #     stratify_cols = interesting_cols_stratification
            #
            #     group_sizes = df.groupby(interesting_cols_stratification).size()
            #     rare_combinations = group_sizes[group_sizes < 2].reset_index()
            #     rare_rows = df.merge(rare_combinations, on=interesting_cols_stratification, how="inner")
            #     print("🔍 Records with too few samples for stratification:")
            #     print(rare_rows)
            #     # stratification not possible on age group, bc to less samples
            #     # stratification also not possible on both labels at the same time I'd say
            #
            #     X_train, X_test, y_train, y_test = train_test_split(
            #         X,
            #         y,
            #         test_size=0.2,
            #         # stratify=y,  # This ensures stratification
            #         stratify=stratify_cols,
            #         random_state=42  # Optional: for reproducibility
            #     )
            #     # Add Split column
            #     X_train = X_train.copy()
            #     X_train["Split"] = "TRAIN"
            #     X_test = X_test.copy()
            #     X_test["Split"] = "TEST"
            #
            #     # Add labels back to X
            #     train_combined = pd.concat([X_train, y_train], axis=1)
            #     test_combined = pd.concat([X_test, y_test], axis=1)
            #
            #     # Combine both into one DataFrame
            #     df = pd.concat([train_combined, test_combined], axis=0).reset_index(drop=True)

            cols_to_check = interesting_cols.copy()
            cols_to_check.append("Split")
            df_check = df[cols_to_check]

            self._analyze_distributions(cols_to_check, df)

            df_train = df_check[df["Split"] == "TRAIN"]
            df_validation = df_check[df["Split"] == "VALIDATION"]
            df_test = df_check[df["Split"] == "TEST"]

            def print_distribution(df, name, cols):
                print(f"\n--- {name.upper()} SPLIT ---")
                self._analyze_distributions(cols, df)

            print_distribution(df_train, "train", cols_to_check)
            print_distribution(df_validation, "validation", cols_to_check)
            print_distribution(df_test, "test", cols_to_check)

    def _analyze_distributions(self, cols_to_check, df):
        for col in cols_to_check:
            counts = df[col].value_counts(dropna=False).sort_index()
            percentages = (counts / counts.sum() * 100).round(2)
            print(f"\nDistribution for '{col}':")
            for value, count, percent in zip(
                counts.index, counts.values, percentages.values
            ):
                print(f"{value}: {count} ({percent}%)")


if __name__ == "__main__":
    # from bias_evaluator import BiasEvaluator
    target_names = ["Eczema", "Fungal", "Others", "Scabies"]
    labels = [0, 1, 2, 3]

    evaluator = BiasEvaluator(target_names=target_names, labels=labels)
    # evaluator.run_full_evaluation(
    #     analysis_name="evaluator standalone",
    #     # add_run_info="big_model",
    #     add_run_info="small_model",
    #     run_detailed_evaluation=True,
    # )

    # evaluator.run_split_distribution_evaluation(test_split=True, create_splits=True)
    evaluator.run_split_distribution_evaluation(test_split=False, create_splits=True)
