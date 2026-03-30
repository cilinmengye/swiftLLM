# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import torch
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
   "Life blooms like a flower, far away",
   "one two three four five",
   "A B C D E F G H I J K L M N O P Q R S T U V",
   "To be or not to be,",
]
# input_ids_batch = [
#     [46811, 76003, 59779, 84592, 118852],
#     [93606, 16554, 91671, 111009],
# ]

# prompts = [
#     {"prompt_token_ids": ids}
#     for ids in input_ids_batch
# ]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0, 
    n=1,
    max_tokens = 100,
)


def main():
    # Create an LLM.
    model_path = "/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B"
    llm = LLM(model=model_path,
              max_model_len=16384)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    started_at = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    runtime = (time.perf_counter() - started_at) * 1000
    print(f"")


    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()