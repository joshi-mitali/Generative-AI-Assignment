"""
src/asklm/crew.py — Assembles the Crew using official decorator pattern and YAML files.
"""

from __future__ import annotations
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew


@CrewBase
class AskLMCrew:
    """AskLM Document Q&A crew built via official YAML-based configuration."""
    
    # Notice that we use relative paths from the current working directory, 
    # but to be safe when running from different locations, we should probably 
    # use paths relative to this file, or let CrewAI's default resolution work.
    # Actually, crewai resolves these relative to where you run it (or the src module).
    # Since we are running app.py in root, config/agents.yaml might not resolve nicely if 
    # CrewAI expects it to be in the module directory. Wait: the official docs show
    # agents_config = "config/agents.yaml". CrewBase resolves relative to the file.
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, llm: LLM, knowledge_sources=None, embedder=None):
        self._llm = llm
        self._knowledge_sources = knowledge_sources or []
        self._embedder = embedder

    @agent
    def document_qa_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["document_qa_specialist"],
            llm=self._llm,
        )

    @task
    def document_qa_task(self) -> Task:
        return Task(
            config=self.tasks_config["document_qa_task"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            knowledge_sources=self._knowledge_sources,
            embedder=self._embedder,
        )
