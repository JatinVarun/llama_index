from unittest.mock import AsyncMock

import pytest

from llama_index.core.evaluation.correctness import CorrectnessEvaluator


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [None, "", "   "])
async def test_correctness_evaluator_rejects_missing_reference(
    reference, mock_llm, monkeypatch
):
    apredict = AsyncMock(return_value="4.0\nCorrect answer.")
    monkeypatch.setattr(type(mock_llm), "apredict", apredict)

    evaluator = CorrectnessEvaluator(llm=mock_llm)

    with pytest.raises(ValueError, match="reference must be provided"):
        await evaluator.aevaluate(
            query="What is Python?",
            response="Python is a programming language.",
            reference=reference,
        )

    apredict.assert_not_awaited()


@pytest.mark.asyncio
async def test_correctness_evaluator_with_valid_reference(mock_llm, monkeypatch):
    apredict = AsyncMock(return_value="4.0\nThe generated answer is correct.")
    monkeypatch.setattr(type(mock_llm), "apredict", apredict)

    evaluator = CorrectnessEvaluator(llm=mock_llm)

    result = await evaluator.aevaluate(
        query="What is Python?",
        response="Python is a programming language.",
        reference="Python is a high-level programming language.",
    )

    assert result.score == 4.0
    assert result.passing is True
    assert result.feedback == "The generated answer is correct."

    apredict.assert_awaited_once()
    assert (
        apredict.await_args.kwargs["reference_answer"]
        == "Python is a high-level programming language."
    )
