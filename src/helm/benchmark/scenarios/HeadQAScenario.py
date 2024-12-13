import json
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


class HeadQAScenario(Scenario):
    """
    From "HEAD-QA: A Healthcare Dataset for Complex Reasoning" (Vilares et al.), HEAD-QA is a multi-choice
    question-answering dataset designed to evaluate reasoning on challenging healthcare-related questions.
    The questions are sourced from Spanish healthcare exams for specialized positions, covering various topics
    such as Medicine, Nursing, Psychology, Chemistry, Pharmacology, and Biology.

    Example from the dataset:

    Question:
    The excitatory postsynaptic potentials:

    A) They are all or nothing.
    B) They are hyperpolarizing.
    C) They can be added.
    D) They spread long distances.
    E) They present a refractory period.

    Answer:
    The answer is C. Explanation: None provided in this dataset.

    @InProceedings{HEAD-QA,
    author = {David Vilares and Manuel Vilares and Carlos Gómez-Rodríguez},
    title = {HEAD-QA: A Healthcare Dataset for Complex Reasoning},
    year = {2019},
    abstract = {We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex reasoning.
    The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging
    even for highly specialized humans. We then consider monolingual (Spanish) and cross-lingual (to English) experiments
    with information retrieval and neural techniques. We show that: (i) HEAD-QA challenges current methods, and (ii) the
    results lag well behind human performance, demonstrating its usefulness as a benchmark for future work.}}


    Task:
    Given a question and its multiple-choice answers, models must identify the correct answer, corresponding to the
    `ra` field in the dataset. The dataset spans six healthcare domains and is challenging even for experts.
    """

    DATASET_DOWNLOAD_URL: str = (
        "https://github.com/HEAD-QA/dataset/raw/main/HEAD-QA.json"
    )

    name = "head_qa"
    description = "A multi-choice QA dataset from Spanish healthcare specialization exams."
    tags = ["question_answering", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Reads the HEAD-QA dataset from the provided JSON file, processes each exam's questions and answers, and
        constructs instances for LLM evaluation.

        :param output_path: Path to save the downloaded JSON file.
        :return: A list of Instance objects containing the question, answers, correct answer, and the split type.
        """
        # Ensure the JSON file is downloaded
        json_path = os.path.join(output_path, "head_qa.json")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=json_path,
            unpack=False,
        )

        instances: List[Instance] = []

        # Load the JSON file
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Process each exam
        for exam_name, exam_data in data["exams"].items():
            for question_data in exam_data["data"]:
                question = question_data["qtext"]
                correct_answer_idx = question_data["ra"]  # Real answer index as a string
                answers = question_data["answers"]

                # Build the options as a formatted string
                options_str = "\n".join(
                    [f"{chr(65 + answer['aid'] - 1)}) {answer['atext']}" for answer in answers]
                )

                # Prepare the input text
                input_text = f"{question}\n\n{options_str}\n\nWhat is the correct answer?"

                # Create references for each answer option
                references = [
                    Reference(
                        Output(text=answer["atext"]),
                        tags=[CORRECT_TAG] if str(answer["aid"]) == correct_answer_idx else [],
                    )
                    for answer in answers
                ]

                # Create an instance
                instance = Instance(
                    input=Input(text=input_text),
                    references=references,
                    split=TRAIN_SPLIT,  # Assuming all data is training data for now
                )

                instances.append(instance)

        return instances
