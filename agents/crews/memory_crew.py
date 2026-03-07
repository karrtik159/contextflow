"""
Memory Crew — background task to extract entities and update the knowledge graph.

Agents:
  - Entity_Extractor: Parses conversation transcripts for facts.
  - Graph_Updater: Persists extracted facts into Neo4j via Mem0.

This crew is invoked as a background task after a voice session ends.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class MemoryCrew:
    """Background memory processor — extracts facts and updates the graph."""

    @agent
    def entity_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["entity_extractor"],
            verbose=True,
            max_iter=10,
        )

    @agent
    def graph_updater(self) -> Agent:
        return Agent(
            config=self.agents_config["graph_updater"],
            verbose=True,
            max_iter=10,
            # TODO: attach Mem0 service tool here
            tools=[],
        )

    @task
    def extract_entities(self) -> Task:
        return Task(config=self.tasks_config["extract_entities"])

    @task
    def update_graph(self) -> Task:
        return Task(
            config=self.tasks_config["update_graph"],
            context=[self.extract_entities()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
