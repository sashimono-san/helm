from typing import Union

from helm.benchmark.adaptation.adapter_spec import ADAPT_GENERATION, AdapterSpec
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec,
    get_multiple_choice_joint_adapter_spec,
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_generation_metric_specs,
    get_exact_match_metric_specs,
    get_generic_metric_specs,
    get_open_ended_generation_metric_specs,
    get_summarization_metric_specs,
    get_basic_metric_specs
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.metrics.metric import MetricSpec


@run_spec_function("head_qa")
def get_head_qa_run_spec(
    language: str = "en", category: Union[str, None] = None
) -> RunSpec:
    """
    RunSpec for the HEAD-QA dataset.
    This configuration evaluates the model's ability to answer challenging multiple-choice biomedical questions.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.headqa_scenario.HeadQAScenario",
        args={"language": language, "category": category},
    )

    # Define the adapter
    adapter_spec = get_multiple_choice_joint_adapter_spec(
        instructions=(
            "You are a highly knowledgeable AI assistant specializing in biomedical sciences. Your task is to answer "
            "multiple-choice questions accurately based on the options provided. Each question will relate to biomedical concepts, "
            "and you will be asked to choose the most appropriate answer.\n\n"
            "Please think step-by-step to solve the question and then generate your final answer. "
            'Before giving the final answer, write "Final Answer: " followed by the corresponding answer alternative.'
        ),
        input_noun="Question",
        output_noun="Answer",
    )

    # Define the metrics
    metric_specs = get_exact_match_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name=f"head_qa:language={language},category={category}",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["biomedical", "head_qa"],
    )


@run_spec_function("medbullets")
def get_medbullets_run_spec() -> RunSpec:
    """
    RunSpec for the MedBullets dataset.
    This configuration evaluates the model's ability to answer challenging multiple-choice clinical questions.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medbullets_scenario.MedBulletsScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_multiple_choice_joint_adapter_spec(
        instructions=(
            "You are a helpful and highly knowledgeable AI assistant specializing in medicine. "
            "Your task is to answer medical questions similar to those found on the USMLE Step 2/3 exams. You will be provided with a clinical scenario, "
            "followed by several multiple-choice options.\n\n"
            "For each question, you must:\n"
            "- Select the correct answer index (A, B, C, D, or E).\n"
            "- Provide the actual answer corresponding to the correct option.\n"
            "- Give a concise explanation for why that answer is correct, based on the clinical scenario provided.\n\n"
            "Please think step-by-step to analyze the clinical scenario before providing your final answer. "
            'Conclude your reasoning with "Final Answer: " followed by the correct answer index and the corresponding answer text.'
        ),
        input_noun="Clinical Scenario",
        output_noun="Answer",
    )

    # Define the metrics
    # get_exact_match_metric_specs - multiple choice exact match (accuracy)
    metric_specs = get_exact_match_metric_specs() + get_generic_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="medbullets",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "medbullets"],
    )


@run_spec_function("medbullets_freetext")
def get_medbullets_freetext_run_spec() -> RunSpec:
    """RunSpec for the MedBullets Free-text dataset."""
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medbullets_scenario.MedBulletsFreeTextScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "You are a helpful and highly knowledgeable AI assistant specializing in medicine. "
            "Your task is to answer medical questions similar to those found on the USMLE Step 2/3 exams. You will be provided with a clinical scenario, "
            "and for each question, you must:\n"
            "- Provide an answer to the question.\n"
            "- Give a concise explanation for why that answer is correct, based on the clinical scenario provided."
        ),
        input_noun="Clinical Scenario",
        output_noun="Answer",
        # NOTE: Setting a high max-tokens yield errors with small models (e.g.: gpt2)
        max_tokens=5000,
    )

    # Define the metrics
    # get_open_ended_generation_metric_specs - ["exact_match", "quasi_exact_match", "f1_score", "rouge_l", "bleu_1", "bleu_4"] bertscore
    metric_specs = get_open_ended_generation_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="medbullets-freetext",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "medbullets-freetext"],
    )


@run_spec_function("aci_bench")
def get_aci_bench_run_spec() -> RunSpec:
    """
    RunSpec for the ACI-Bench dataset.
    This configuration evaluates the model's ability to summarize doctor-patient dialogues into structured clinical notes.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.aci_bench_scenario.ACIBenchScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "Summarize the conversation to generate a clinical note with four sections:\n"
            "1. HISTORY OF PRESENT ILLNESS\n"
            "2. PHYSICAL EXAM\n"
            "3. RESULTS\n"
            "4. ASSESSMENT AND PLAN\n\n"
        ),
        max_tokens=5000,  # Limit token count to ensure concise output
        input_noun="Conversation",
        output_noun="Clinical Note",
        temperature=0.7,  # Allow for slight variability in generation
    )

    # Define the metrics
    # summarization metric maybe useful - get_summarization_metric_specs
    metric_specs = get_open_ended_generation_metric_specs()

    # Return the RunSpec
    return RunSpec(
        name="aci_bench",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "aci_bench"],
    )


@run_spec_function("medec")
def get_medec_run_spec() -> RunSpec:
    """
    RunSpec for the MEDEC dataset.
    This configuration evaluates the model's ability to summarize doctor-patient dialogues into structured clinical notes.
    """
    # Define the scenario
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.medec_scenario.MedecScenario",
        args={},
    )

    # Define the adapter
    adapter_spec = get_generation_adapter_spec(
        instructions=(
            "The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. "
            "The text is either correct or contains one error. The text has a sentence per line. Each line starts with the "
            "sentence ID, followed by a space character then the sentence to check. Check every sentence of the text. "
            "If the text is correct return the following output: CORRECT. If the text has a medical error, return the "
            "sentence ID of the sentence containing the error, followed by a space, and a corrected version of the sentence."
        ),
        max_tokens=5000,  # Limit token count to ensure concise output
        input_noun="Conversation",
        output_noun="Clinical Note",
        max_train_instances = 10,
        num_outputs = 10
    )

    # Define the metrics
    metric_specs = [
        MetricSpec(
            class_name="helm.benchmark.metrics.medec_metrics.MedecMetric",
            args={},
        )
    ] + get_basic_metric_specs([])

    # Return the RunSpec
    return RunSpec(
        name="medec",
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["clinical", "medec"],
    )
