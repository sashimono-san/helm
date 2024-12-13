import csv
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    Input,
    Output,
)
from helm.common.general import ensure_file_downloaded


class ACIBenchScenario(Scenario):
    """
    From "Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation"
    (Yim et al.), ACI-Bench is the largest dataset to date tackling the problem of AI-assisted note generation from
    doctor-patient dialogue. This dataset enables benchmarking and evaluation of generative models, focusing on the
    arduous task of converting clinical dialogue into structured electronic medical records (EMR).

    Example from the dataset:

    Dialogue:
    [doctor] hi, brian. how are you?
    [patient] hi, good to see you.
    [doctor] it's good to see you too. so, i know the nurse told you a little bit about dax.
    [patient] mm-hmm.
    [doctor] i'd like to tell dax about you, okay?
    [patient] sure.

    Note:
    CHIEF COMPLAINT

    Follow-up of chronic problems.

    HISTORY OF PRESENT ILLNESS

    @Article{ACI-Bench,
    author = {Wen-wai Yim, Yujuan Fu, Asma Ben Abacha, Neal Snider, Thomas Lin, Meliha Yetisgen},
    title = {Aci-bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
    journal = {Nature Scientific Data},
    year = {2023},
    abstract = {Recent immense breakthroughs in generative models such as in GPT4 have precipitated re-imagined ubiquitous
    usage of these models in all applications. One area that can benefit by improvements in artificial intelligence (AI)
    is healthcare. The note generation task from doctor-patient encounters, and its associated electronic medical record
    documentation, is one of the most arduous time-consuming tasks for physicians. It is also a natural prime potential
    beneficiary to advances in generative models. However with such advances, benchmarking is more critical than ever.
    Whether studying model weaknesses or developing new evaluation metrics, shared open datasets are an imperative part
    of understanding the current state-of-the-art. Unfortunately as clinic encounter conversations are not routinely
    recorded and are difficult to ethically share due to patient confidentiality, there are no sufficiently large clinic
    dialogue-note datasets to benchmark this task. Here we present the Ambient Clinical Intelligence Benchmark (aci-bench)
    corpus, the largest dataset to date tackling the problem of AI-assisted note generation from visit dialogue. We also
    present the benchmark performances of several common state-of-the-art approaches.}}

    Task:
    Given a doctor-patient dialogue, models must generate a clinical note that summarizes the conversation,
    focusing on the chief complaint, history of present illness, and other relevant clinical information.
    """

    DATASET_DOWNLOAD_URL: str = (
        "https://github.com/wyim/aci-bench/raw/main/data/challenge_data/valid.csv"
    )

    name = "aci_bench"
    description = (
        "The Ambient Clinical Intelligence Benchmark (ACI Bench) corpus evaluates the task of "
        "generating medical notes from doctor-patient dialogue."
    )
    tags = ["summarization", "medicine"]

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Reads the ACI-Bench dataset from the provided CSV file, processes each dialogue-note pair, and
        creates instances for evaluation. The task involves summarizing the dialogue into a clinical note.

        :param output_path: Path to save the downloaded CSV file.
        :return: A list of Instance objects containing the dialogue, ground truth note, and the split type.
        """
        # Ensure the CSV file is downloaded
        csv_path = os.path.join(output_path, "aci_bench_valid.csv")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=csv_path,
            unpack=False,
        )

        instances: List[Instance] = []

        # Read the CSV file and process each row
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                dialogue = row["dialogue"]
                note = row["note"]

                # Prepare the input text (dialogue)
                input_text = f"Summarize the following doctor-patient dialogue into a clinical note:\n\n{dialogue}"

                # Create an instance with the ground truth (note)
                instance = Instance(
                    input=Input(text=input_text),
                    references=[Reference(Output(text=note), tags=[CORRECT_TAG])],
                    split=TRAIN_SPLIT,  # Assuming this dataset is for training
                )

                instances.append(instance)

        return instances
