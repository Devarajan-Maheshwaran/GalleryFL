import json
from pathlib import Path

from taxonomy_parser import TaxonomyParser


def test_report_contract_if_evaluated():
    report_path = Path("output/retrain/test_metrics.json")
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    parser = TaxonomyParser("taxonomy.json")
    assert report["task"] == "single_label_multiclass"
    assert report["decision_rule"] == "argmax"
    assert report["labels"] == parser.model_labels
    assert set(report["per_class"]) == set(parser.model_labels)
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert 0.0 <= report["accuracy"] <= 1.0
