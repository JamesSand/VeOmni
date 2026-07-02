# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QADConfig argument plumbing: dataclass defaults, validation, YAML parsing."""

import sys

import pytest
import yaml

from veomni.arguments import QADConfig, VeOmniArguments, parse_args


class TestQADConfig:
    def test_defaults(self):
        cfg = QADConfig()
        assert cfg.enable is False
        assert cfg.mode == "w4a4"
        assert cfg.teacher_mode == "separate"
        assert "q_proj" in cfg.target_modules and "down_proj" in cfg.target_modules

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mode": "w8"}, "train.qad.mode"),
            ({"teacher_mode": "toggle"}, "train.qad.teacher_mode"),
            ({"alpha": 1.5}, "alpha"),
            ({"tau": 0.0}, "tau"),
            ({"teacher_topk": 0}, "teacher_topk"),
            ({"enable": True, "mode": "w4a4", "calib_steps": 0}, "calib_steps"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            QADConfig(**kwargs)

    def test_w4_does_not_require_calib(self):
        assert QADConfig(enable=True, mode="w4", calib_steps=0).mode == "w4"


class TestYamlParsing:
    def test_train_qad_yaml_roundtrip(self, tmp_path, monkeypatch):
        config = {
            "model": {"config_path": "tests/toy_config/qwen2_toy.json"},
            "data": {"train_path": str(tmp_path)},
            "train": {
                "checkpoint": {"output_dir": str(tmp_path / "out")},
                "qad": {
                    "enable": True,
                    "mode": "w4",
                    "tau": 2.0,
                    "alpha": 0.9,
                    "teacher_topk": 64,
                    "target_modules": ["q_proj", "o_proj"],
                },
            },
        }
        config_file = tmp_path / "qad.yaml"
        config_file.write_text(yaml.dump(config))
        monkeypatch.setattr(sys, "argv", ["prog", str(config_file)])
        args = parse_args(VeOmniArguments)
        assert args.train.qad.enable is True
        assert args.train.qad.mode == "w4"
        assert args.train.qad.tau == 2.0
        assert args.train.qad.alpha == 0.9
        assert args.train.qad.teacher_topk == 64
        assert args.train.qad.target_modules == ["q_proj", "o_proj"]
        # untouched fields keep defaults
        assert args.train.qad.teacher_mode == "separate"

    def test_cli_override(self, tmp_path, monkeypatch):
        config = {
            "model": {"config_path": "tests/toy_config/qwen2_toy.json"},
            "data": {"train_path": str(tmp_path)},
            "train": {"checkpoint": {"output_dir": str(tmp_path / "out")}},
        }
        config_file = tmp_path / "base.yaml"
        config_file.write_text(yaml.dump(config))
        monkeypatch.setattr(
            sys, "argv", ["prog", str(config_file), "--train.qad.enable", "true", "--train.qad.mode", "a4"]
        )
        args = parse_args(VeOmniArguments)
        assert args.train.qad.enable is True
        assert args.train.qad.mode == "a4"
