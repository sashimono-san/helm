import json
import os
from typing import List, Dict
from helm.common.general import ensure_file_downloaded
from helm.common.constants import TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG
from helm.benchmark.scenarios.scenario import Scenario, Instance, Reference, Input, Output


class MedBulletScenario(Scenario):
    """
    From "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions"
    (Chen et al.), MedBullet is a dataset comprising USMLE Step 2&3 style clinical questions. The dataset
    is designed to evaluate the performance of LLMs in answering and explaining challenging medical questions,
    emphasizing the need for explainable AI in medical QA.

    Example from the dataset:

    Question:
    A 42-year-old woman is enrolled in a randomized controlled trial to study cardiac function in the setting of
    several different drugs. She is started on verapamil and instructed to exercise at 50% of her VO2 max while
    several cardiac parameters are being measured. During this experiment, which of the following represents
    the relative conduction speed through the heart from fastest to slowest?

    A) AV node > ventricles > atria > Purkinje fibers
    B) Purkinje fibers > ventricles > atria > AV node
    C) Purkinje fibers > atria > ventricles > AV node
    D) Purkinje fibers > AV node > ventricles > atria

    Answer:
    The answer is C. Explanation: The conduction velocity of the structures of the heart is in the following order:
    Purkinje fibers > atria > ventricles > AV node. A calcium channel blocker such as verapamil would only slow
    conduction in the AV node.

    @Article{MedBullet,
    author = {Hanjie Chen and Zhouxiang Fang and Yash Singla and Mark Dredze},
    title = {Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions},
    year = {2023},
    abstract = {LLMs have demonstrated impressive performance in answering medical questions, such as passing scores
    on medical licensing examinations. However, medical board exam questions or general clinical questions do not
    capture the complexity of realistic clinical cases. Moreover, the lack of reference explanations means we cannot
    easily evaluate the reasoning of model decisions, a crucial component of supporting doctors in making complex
    medical decisions. To address these challenges, we construct two new datasets: JAMA Clinical Challenge and
    Medbullets. JAMA Clinical Challenge consists of questions based on challenging clinical cases, while Medbullets
    comprises USMLE Step 2&3 style clinical questions. Both datasets are structured as multiple-choice question-answering
    tasks, where each question is accompanied by an expert-written explanation. We evaluate four LLMs on the two
    datasets using various prompts. Experiments demonstrate that our datasets are harder than previous benchmarks.
    The inconsistency between automatic and human evaluations of model-generated explanations highlights the need
    to develop new metrics to support future research on explainable medical QA.}}

    Task:
    Given a clinical question with multiple-choice options, models must identify the correct answer and generate a
    response that includes the reasoning, as described in the expert-written explanation.
    """

    # TODO: Add a base url
    DATASET_DOWNLOAD_BASE_URL: str = "https://github.com/HanjieChen/ChallengeClinicalQA/tree/main/medbullets"

    name = "medbullet"
    description = "USMLE Step 2&3 style clinical questions with explanations."
    tags = ["reasoning", "biomedical"]

    def get_instances(self, output_path: str) -> List[Instance]:
        splits = {"_op4": TRAIN_SPLIT, "_op5": TEST_SPLIT}
        instances: List[Instance] = []

        for split, split_tag in splits.items():  # Iterate over the splits
            #source_url: str = f"{self.DATASET_DOWNLOAD_BASE_URL}/{split}.jsonl"
            #data_path: str = os.path.join(output_path, f"med_calc_bench_{split}")
            #ensure_file_downloaded(source_url=source_url, target_path=data_path)
            #        csv_path = os.path.join(output_path, "medbullets.csv")
            
            csv_path = os.path.join(output_path, f"medbullets_{split}.csv")
            ensure_file_downloaded(
                source_url=self.DATASET_DOWNLOAD_BASE_URL,
                target_path=csv_path,
                unpack=False,
            )

            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    example: Dict = json.loads(line.strip())

                    # Map from answer_idx to the corresponding option text
                    option_map = {
                        "A": example.get("opa", ""),
                        "B": example.get("opb", ""),
                        "C": example.get("opc", ""),
                        "D": example.get("opd", ""),
                        "E": example.get("ope", ""),
                    }

                    references = [
                        Reference(
                            Output(text=option_text),
                            tags=[CORRECT_TAG] if option == example["answer_idx"] else [],
                        )
                        for option, option_text in option_map.items()
                    ]

                    instances.append(
                        Instance(
                            input=Input(text=example["question"]),
                            references=references,
                            split=split_tag,
                        )
                    )

        return instances