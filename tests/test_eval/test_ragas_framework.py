from pathlib import Path

from app.evals.ragas_framework import (
    EvalCase,
    build_evaluation_dataset,
    load_eval_cases,
    materialize_responses,
    select_metric_names,
)


def test_load_eval_cases_reads_fixture():
    cases = load_eval_cases(Path("tests/test_eval/fixtures/ragas_cases.json"))

    assert len(cases) == 10
    assert cases[0].user_input == "How does the voice worker reach the backend?"
    assert cases[0].response is None


def test_select_metrics_uses_available_fields():
    answer_only = [
        EvalCase(user_input="q1", response="a1"),
    ]
    answer_with_reference = [
        EvalCase(user_input="q1", response="a1", reference="r1"),
    ]
    full_rag = [
        EvalCase(
            user_input="q1",
            response="a1",
            reference="r1",
            retrieved_contexts=["ctx"],
        ),
    ]

    assert select_metric_names(answer_only) == ["answer_relevancy"]
    assert select_metric_names(answer_with_reference) == [
        "answer_relevancy",
        "answer_correctness",
    ]
    assert select_metric_names(full_rag) == [
        "answer_relevancy",
        "answer_correctness",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]


def test_materialize_responses_uses_target_for_missing_answers():
    class FakeTarget:
        def answer(self, user_input: str, user_id: str) -> str:
            return f"{user_id}:{user_input}"

    hydrated = materialize_responses(
        [
            EvalCase(user_input="q1", user_id="u1"),
            EvalCase(user_input="q2", user_id="u2", response="existing"),
        ],
        FakeTarget(),
    )

    assert hydrated[0].response == "u1:q1"
    assert hydrated[1].response == "existing"


def test_build_evaluation_dataset_serialises_cases():
    dataset = build_evaluation_dataset(
        [
            EvalCase(
                user_input="q1",
                response="a1",
                reference="r1",
                retrieved_contexts=["ctx1"],
            )
        ]
    )

    assert dataset.to_list() == [
        {
            "user_input": "q1",
            "response": "a1",
            "reference": "r1",
            "retrieved_contexts": ["ctx1"],
        }
    ]
