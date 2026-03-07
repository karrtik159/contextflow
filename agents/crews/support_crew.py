"""
Support Crew — handles complex knowledge queries.

Agents:
  - Context_Gatherer: Queries pgvector + Neo4j in parallel.
  - Answer_Synthesizer: Generates a personalized response from context.

This crew is invoked by the RAG Voice Agent when a user asks a
knowledge-intensive question.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


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
            # TODO: attach pgvector + Neo4j search tools here
            tools=[],
        )

    @agent
    def answer_synthesizer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer_synthesizer"],
            verbose=True,
            memory=True,
            max_iter=10,
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
