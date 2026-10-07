import json
import sys

import pytest
from run_experiment import main


def test_unsupported_calibration_grid_is_rejected_before_any_fitting(tmp_path, monkeypatch, capsys):
    config = tmp_path / 'config.yaml'
    config.write_text(json.dumps({'sample_sizes': [36, 107]}))
    monkeypatch.setattr(sys, 'argv', ['run_experiment.py', '--config', str(config), '--stage', 'pilot'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert '71' in capsys.readouterr().err
