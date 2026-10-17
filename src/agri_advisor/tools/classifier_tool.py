"""LangChain tool exposing the image classifier to the agent."""

from pathlib import Path

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agri_advisor.services.errors import ServiceError
from agri_advisor.vision.decision import ClassifierOutput, DecisionStatus
from agri_advisor.vision.images import resolve_image
from agri_advisor.vision.service import DiagnosisService

_GUIDANCE = {
    DecisionStatus.CONFIDENT: (
        "Guidance: one candidate clearly leads. Present it as the most likely, mention the "
        "runner-up briefly, say it is a model estimate, and ask the user to compare it "
        "with the symptoms they actually see."
    ),
    DecisionStatus.AMBIGUOUS: (
        "Guidance: the top two candidates are close. Do not pick one yet. Ask whether the "
        "plants are in a greenhouse or an open field, the region and date, and which symptom "
        "details (lesion shape, colour, position on the plant) distinguish them. Recent "
        "weather can make one more plausible but never confirms it."
    ),
    DecisionStatus.UNCERTAIN: (
        "Guidance: no clear candidate. Ask for a clearer, closer photo of one affected leaf "
        "in daylight, and for more symptom details. Do not name a disease as likely."
    ),
}


class ClassifyImageArgs(BaseModel):
    """Arguments of the classify_leaf_image tool."""

    image_id: str = Field(
        description="The image id from the '[image attached: <id>]' note in the user's message."
    )


def format_classifier_output(output: ClassifierOutput) -> str:
    """Render the classifier output as compact text for the LLM."""
    lines = ["Image classifier result (model estimate, not a confirmed diagnosis):"]
    for rank, prediction in enumerate(output.top_k, start=1):
        lines.append(f"{rank}. {prediction.label}: {prediction.probability:.0%}")
    lines.append(f"Margin between the top two: {output.margin:.2f}")
    lines.append(f"Status: {output.status.value}")
    lines.append(_GUIDANCE[output.status])
    return "\n".join(lines)


def build_classify_tool(
    service: DiagnosisService,
    uploads_dir: Path,
    notice: str | None = None,
) -> BaseTool:
    """Create the classifier tool. `notice` is prepended to every result (e.g. a stub warning)."""

    @tool("classify_leaf_image", args_schema=ClassifyImageArgs)
    def classify_leaf_image(image_id: str) -> str:
        """Classify a tomato plant photo that the user attached, using the image classifier.

        Use it when the user's message contains an '[image attached: <id>]' note and pass
        exactly that id. Returns the top candidate diseases with probabilities and a status.
        The result is a model estimate, never a confirmed diagnosis.
        """
        try:
            path = resolve_image(image_id, uploads_dir)
            output = service.diagnose(path)
        except ServiceError as exc:
            return f"Image classification failed: {exc}"
        text = format_classifier_output(output)
        return f"{notice}\n{text}" if notice else text

    return classify_leaf_image