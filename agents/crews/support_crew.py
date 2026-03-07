"""
Support Crew — handles complex knowledge queries via Hybrid Graph-Vector RAG.

Agents:
  - Context_Gatherer: Queries pgvector + Neo4j + user memories in parallel.
  - Answer_Synthesizer: Generates a personalized response from context.

Usage:
    result = SupportCrew().crew().kickoff(inputs={"query": "...", "user_id": "..."})
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from agents.crews.tools.graph_search_tool import GraphSearchTool
from agents.crews.tools.mem0_tool import MemorySearchTool
from agents.crews.tools.vector_search_tool import VectorSearchTool


@CrewBase
class SupportCrew:
    """Hybrid RAG crew — retrieves context and synthesizes answers."""

    @agent
    def context_gatherer(self) -> Agent:
        return Agent(
            config=self.agents_config["context_gatherer"],
            verbose=True,
            memory=True,
            max_iter=10,
            tools=[
                VectorSearchTool(),
                GraphSearchTool(),
                MemorySearchTool(),
            ],
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],
            verbose=True,
            memory=True,
            max_iter=10,
            tools=[MemorySearchTool()],  # Can look up user prefs while writing
        )

    @task
    def retrieve_context(self) -> Task:
        return Task(config=self.tasks_config["retrieve_context"])

    @task
    def synthesize_answer(self) -> Task:
        return Task(
            config=self.tasks_config["synthesize_answer"],
            context=[self.retrieve_context()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
        )
