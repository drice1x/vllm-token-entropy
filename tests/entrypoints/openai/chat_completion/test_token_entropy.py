from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def test_return_entropy_reaches_sampling_params() -> None:
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        return_entropy=True,
    )

    params = request.to_sampling_params(max_tokens=4, default_sampling_params={})

    assert params.return_entropy is True
