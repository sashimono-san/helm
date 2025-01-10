from typing import Union

from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_multiple_choice_joint_adapter_spec,
    get_generation_adapter_spec
)
from helm.benchmark.metrics.common_metric_specs import (
    get_open_ended_generation_metric_specs
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.metrics.metric import MetricSpec

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
            "The conversation is:"
        ),
        input_noun="Conversation",
        output_noun="Clinical Note",
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
