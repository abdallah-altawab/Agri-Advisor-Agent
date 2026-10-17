from pathlib import Path

import pytest

from agri_advisor.tools.classifier_tool import build_classify_tool
from agri_advisor.vision.images import store_image
from agri_advisor.vision.service import DiagnosisService
from agri_advisor.vision.stub import StubClassifier


@pytest.mark.parametrize("scenario", ["confident", "ambiguous", "uncertain"])
def test_tool_reports_status_and_ranked_candidates(scenario, make_image, uploads_dir):
    image_id = store_image(make_image(), uploads_dir)
    service = DiagnosisService(StubClassifier.from_scenario(scenario))
    tool = build_classify_tool(service, uploads_dir)

    output = tool.invoke({"image_id": image_id})

    assert f"Status: {scenario}" in output
    assert "1. candidate_A" in output
    assert "Guidance:" in output


def test_tool_prepends_notice_when_given(make_image, uploads_dir):
    image_id = store_image(make_image(), uploads_dir)
    service = DiagnosisService(StubClassifier.from_scenario("ambiguous"))
    tool = build_classify_tool(service, uploads_dir, notice="DEMO NOTICE")

    assert tool.invoke({"image_id": image_id}).startswith("DEMO NOTICE")


def test_tool_rejects_unsafe_image_ids(uploads_dir):
    service = DiagnosisService(StubClassifier.from_scenario("confident"))
    tool = build_classify_tool(service, uploads_dir)

    output = tool.invoke({"image_id": "../etc/passwd"})

    assert output.startswith("Image classification failed:")
    assert "Invalid image id" in output


def test_service_applies_temperature():
    logits = [3.0, 1.0, 0.0, 0.0]
    sharp = DiagnosisService(StubClassifier(logits), temperature=1.0).diagnose(Path("x.jpg"))
    flat = DiagnosisService(StubClassifier(logits), temperature=5.0).diagnose(Path("x.jpg"))
    assert flat.top_k[0].probability < sharp.top_k[0].probability


def test_service_rejects_wrong_logit_count():
    classifier = StubClassifier([1.0, 2.0, 3.0])  # 3 logits but 4 default class names
    with pytest.raises(ValueError, match="logits"):
        DiagnosisService(classifier).diagnose(Path("x.jpg"))