# tests/geomemvla/test_dataloader_episode_fields.py
# [Geo-MemoryVLA] Guards that the LIBERO sample dict carries episode_id + timestep,
# required by the dual memory bank.
import ast
import pathlib


def test_sample_dict_includes_episode_and_timestep():
    src = pathlib.Path("starVLA/dataloader/gr00t_lerobot/datasets.py").read_text()
    # The sample dict literal must reference these keys.
    assert '"episode_id"' in src, "episode_id not added to sample dict"
    assert '"timestep"' in src, "timestep not added to sample dict"
    # And it must be parseable Python (no syntax break from the edit).
    ast.parse(src)
