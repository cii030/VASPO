"""CPU checks for the imported critic pipeline without downloading models."""

import ast
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CRITIC = ROOT / 'critic_init'


@pytest.mark.parametrize('nodes', [1, 2])
def test_critic_launch_from_another_directory(tmp_path, nodes):
    stub = tmp_path / 'torchrun'
    stub.write_text('#!/bin/sh\nexec "' + sys.executable + '" -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' "$@"\n')
    stub.chmod(0o755)
    train = tmp_path / 'critic train.jsonl'
    train.write_text('{}\n')
    env = dict(os.environ, PATH=str(tmp_path)+':'+os.environ['PATH'],
               MODEL_PATH=str(tmp_path/'actor model'), TRAIN_FILE=str(train),
               OUTPUT_DIR=str(tmp_path/'critic output'), NNODES=str(nodes))
    if nodes > 1:
        env.update(NODE_RANK='1', MASTER_ADDR='127.0.0.1', NGPUS='2',
                   GRADIENT_ACCUMULATION_STEPS='16')
    result = subprocess.run(
        ['bash', str(CRITIC/'scripts/train.sh'), '--max_steps', '1'],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=True,
    )
    args = json.loads(result.stdout)
    assert f'--nnodes={nodes}' in args
    assert str(CRITIC/'runner/train.py') in args
    assert args[args.index('--model_name_or_path')+1] == str(tmp_path/'actor model')
    assert args[args.index('--train_file')+1] == str(train)
    assert args[args.index('--output_dir')+1] == str(tmp_path/'critic output')
    assert args[args.index('--num_train_epochs')+1] == '2'
    assert args[args.index('--learning_rate')+1] == '5e-6'
    assert '--bf16' in args
    config = Path(args[args.index('--deepspeed')+1])
    assert config == CRITIC/'configs/zero1.json'
    assert json.loads(config.read_text())['zero_optimization']['stage'] == 1
    assert args[-2:] == ['--max_steps', '1']
    if nodes == 1:
        assert '--nproc_per_node=8' in args
        assert args[args.index('--gradient_accumulation_steps')+1] == '256'
    else:
        assert '--node_rank=1' in args and '--master_addr=127.0.0.1' in args
        assert '--nproc_per_node=2' in args
        assert args[args.index('--gradient_accumulation_steps')+1] == '16'


def test_response_mask_uses_final_response_occurrence():
    module = runpy.run_path(str(CRITIC/'runner/response_mask.py'))
    prompt = 'user repeats answer 2\nassistant: answer 2<|im_end|>'
    response = 'answer 2'
    start, end = module['find_response_char_span'](prompt, response)
    assert start == prompt.rfind(response)
    offsets = [(0, 0), *[(i, i+1) for i in range(len(prompt))]]
    mask = module['mask_from_offsets'](offsets, start, end)
    assert sum(mask) == len(response) and mask[0] == 0
    assert mask[1:1+start] == [0] * start
    assert mask[1+start:1+end] == [1] * len(response)
    assert not any(mask[1+end:])


def test_critic_loss_shift_and_padding():
    # Execute the actual class without importing the optional GPU training stack.
    path = CRITIC/'runner/pretrainer.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    namespace = {'torch': torch, 'Trainer': object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    trainer = object.__new__(namespace['PreTrainer'])

    class ValueModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(torch.zeros(1, 4))

        def forward(self, **kwargs):
            return {'logits': self.values.unsqueeze(-1)}

    model = ValueModel().eval()
    loss = trainer.compute_loss(model, {
        'labels': torch.ones(1, 4),
        'response_mask': torch.tensor([[0, 0, 1, 1]]),
        'attention_mask': torch.tensor([[1, 1, 1, 0]]),
    })
    torch.testing.assert_close(loss, torch.tensor(.25))
    loss.backward()
    # Only position 1 predicts a supervised response action. Prompt-only and
    # padded/final positions must not receive a loss gradient.
    torch.testing.assert_close(model.values.grad, torch.tensor([[0., -.25, 0., 0.]]))


@pytest.mark.parametrize('mode', ['raw', 'answer', 'solution'])
def test_critic_converter_cli(tmp_path, mode):
    # Stub only the external tokenizer; exercise the real parser, readers,
    # reference matching, balancing, prompt construction, and JSONL writer.
    (tmp_path/'transformers.py').write_text('''
class AutoTokenizer:
    chat_template = 'test'
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()
    def apply_chat_template(self, messages, **kwargs):
        return '\\n'.join(message['role'] + ': ' + message['content'] for message in messages)
''')
    rollouts = tmp_path/'rollouts.jsonl'
    rows = [
        {'question': 'What is 1 + 1?', 'answer': '2', 'response': text, 'label': label}
        for text, label in [('2.', 1), ('Two.', 1), ('The answer is 2.', 1), ('3.', 0)]
    ]
    rollouts.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    references = tmp_path/'references.jsonl'
    references.write_text(json.dumps({
        'question': 'What is 1 + 1?',
        'solution': ['One plus one equals two.', 'Unused alternate derivation.'],
    })+'\n')
    output = tmp_path/'critic.jsonl'
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    subprocess.run([
        sys.executable, str(CRITIC/'scripts/convert_infer_to_critic_jsonl.py'),
        '--mode', mode, '--input_file', str(rollouts), '--output_file', str(output),
        '--reference_file', str(references), '--model_name_or_path', 'test-tokenizer',
        '--multi_solution_mode', 'first', '--all_correct_downsample_factor', '1',
        '--balance', 'upsample',
    ], env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    converted = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(converted) == 6
    assert Counter(row['label'] for row in converted) == {1.0: 3, 0.0: 3}
    for row in converted:
        assert row['response'] in row['prompt']
        assert ('The ground truth answer is 2.' in row['prompt']) == (mode != 'raw')
        assert ('One plus one equals two.' in row['prompt']) == (mode == 'solution')
        assert 'Unused alternate derivation.' not in row['prompt']
