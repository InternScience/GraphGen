
# Output Formats
we support generating datasets in alpaca, sharegpt and chatml format.

## Alpaca Format
### Supervised Fine-Tuning Dataset
- [Example](resources/output_examples/atomic_alpaca.json)
In supervised fine-tuning, the `instruction` column will be concatenated with the `input` column and used as the user prompt, then the user prompt would be `instruction\ninput`. The `output` column represents the model response.
```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)"
  }
]
```

## Sharegpt Format
### Supervised Fine-Tuning Dataset
- [Example](resources/output_examples/cot_sharegpt.json)
Compared to the alpaca format, the sharegpt format allows the datasets have more roles, such as human, gpt, observation and function. They are presented in a list of objects in the `conversations` column.

Note that the human and observation should appear in odd positions, while gpt and function should appear in even positions. The gpt and function will be learned by the model.

In our implementation, only `human` and `gpt` will be used.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction (required)"
      },
      {
        "from": "gpt",
        "value": "model response (required)"
      }
    ]
    }
]
```

## ChatML Format
### Supervised Fine-Tuning Dataset
- [Example](resources/output_examples/aggregated_chatml.json)
Like the sharegpt format, the chatml format also allows the datasets have more roles, such as user, assistant, system and tool. They are presented in a list of objects in the `messages` column.

In our implementation, only `user` and `assistant` will be used.

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "user instruction (required)"
      },
      {
        "role": "assistant",
        "content": "model response (required)"
      }
    ]
    }
]
```
