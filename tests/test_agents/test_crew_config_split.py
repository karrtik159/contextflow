from pathlib import Path

import yaml


def _load_yaml(relative_path: str) -> dict:
    path = Path(relative_path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_support_crew_tasks_only_reference_support_agents():
    agents = _load_yaml("agents/crews/config/support_agents.yaml")
    tasks = _load_yaml("agents/crews/config/support_tasks.yaml")

    referenced_agents = {task_config["agent"] for task_config in tasks.values()}
    assert referenced_agents == {"context_gatherer", "answer_synthesizer"}
    assert referenced_agents.issubset(set(agents))


def test_memory_crew_tasks_only_reference_memory_agents():
    agents = _load_yaml("agents/crews/config/memory_agents.yaml")
    tasks = _load_yaml("agents/crews/config/memory_tasks.yaml")

    referenced_agents = {task_config["agent"] for task_config in tasks.values()}
    assert referenced_agents == {"entity_extractor", "graph_updater"}
    assert referenced_agents.issubset(set(agents))
