"""Task Router for selecting task-specific attention heads.

Maps API requests to appropriate TaskSpecificAttentionHead based on
task type parameter.
"""

from __future__ import annotations

from typing import Any, Dict

from core.task_attention import create_task_head, TASK_REGISTRY


class TaskRouter:
    """Routes requests to task-specific attention heads.
    
    Maintains a cache of initialized task heads for efficient reuse.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for task heads.
    output_dim : int
        Output dimension for task heads.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 256):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._task_heads: Dict[str, Any] = {}
        self._init_all_heads()

    def _init_all_heads(self) -> None:
        """Initialize all registered task heads."""
        for task_name in TASK_REGISTRY.keys():
            self._task_heads[task_name] = create_task_head(
                task_name=task_name,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
            )

    def get_task_head(self, task_name: str) -> Any:
        """Get the task-specific attention head.
        
        Parameters
        ----------
        task_name : str
            Task type identifier.
            
        Returns
        -------
        TaskSpecificAttentionHead
            The task head for the specified task.
            
        Raises
        ------
        ValueError
            If task_name is not registered.
        """
        if task_name not in self._task_heads:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(TASK_REGISTRY.keys())}"
            )
        return self._task_heads[task_name]

    def list_available_tasks(self) -> list[str]:
        """Return list of available task names."""
        return list(self._task_heads.keys())
