from helm.benchmark.adaptation.adapter_spec import (
    ADAPT_GENERATION,
    AdapterSpec,
)
from helm.benchmark.adaptation.common_adapter_specs import (
    get_generation_adapter_spec
)
from helm.benchmark.metrics.common_metric_specs import (
    get_basic_metric_specs
)
from helm.benchmark.run_spec import RunSpec, run_spec_function
from helm.benchmark.scenarios.scenario import ScenarioSpec
from helm.benchmark.metrics.metric import MetricSpec


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
        #max_tokens=2000,  # Limit token count to ensure concise output
        input_noun="Conversation",
        output_noun="Clinical Note",
        max_train_instances = 2,
        num_outputs = 2
        #temperature=0.7,  # Allow for slight variability in generation
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
