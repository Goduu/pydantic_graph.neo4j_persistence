"""
Implementation of BaseStatePersistence for Neo4j.
"""
from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Optional, Union
from uuid import UUID, uuid4
import json
from datetime import datetime
import asyncio

import pydantic
from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import (
    BaseStatePersistence, NodeSnapshot, Snapshot, EndSnapshot, SnapshotStatus,
    build_snapshot_list_type_adapter
)
from pydantic_graph.exceptions import GraphNodeStatusError
from pydantic_graph.nodes import StateT, RunEndT
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, UserPromptPart, SystemPromptPart, RetryPromptPart,
    TextPart, ToolCallPart, ToolReturnPart, ModelResponsePart, ModelRequestPart, ModelMessage
)
from neo4j import AsyncGraphDatabase

def serialize_complex_object(obj):
    """Serialize complex objects including datetime and custom classes."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Special handling for ModelMessage objects from pydantic_ai
    elif hasattr(obj, 'model_dump_json') and callable(getattr(obj, 'model_dump_json')):
        try:
            # Use Pydantic's serialization for objects that support it
            return json.loads(obj.model_dump_json())
        except Exception:
            # Fallback if model_dump_json fails
            pass
    elif hasattr(obj, '__dict__'):
        # Convert object to dict and recursively serialize its values
        return {k: serialize_complex_object(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, list):
        # Handle lists by recursively serializing each item
        return [serialize_complex_object(item) for item in obj]
    elif isinstance(obj, dict):
        # Handle dictionaries by recursively serializing values
        return {k: serialize_complex_object(v) for k, v in obj.items()}
    return obj


@dataclass
class Neo4jStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """Neo4j based state persistence that hold graph run state in a Neo4j database."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        execution_id: Optional[UUID] = None,
        verbose: bool = False
    ):
        """
            Initialize Neo4j persistence handler.

            Args:
                uri: Neo4j connection URI
                username: Neo4j username
                password: Neo4j password
                execution_id: Optional ID for this execution, will generate UUID if not provided
        """
        if verbose:
            print(
                f"[Neo4jStatePersistence.__init__] Initializing with, execution_id={execution_id}")
        self.async_driver = AsyncGraphDatabase.driver(
            uri, auth=(username, password))
        self.execution_id = execution_id or uuid4()
        self._types_set = False
        self._state_type = None
        self._run_end_type = None
        self.verbose = verbose
    _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
        default=None, init=False, repr=False
    )

    async def close(self):
        """Close the Neo4j drivers properly."""
        if self.verbose:
            print("[Neo4jStatePersistence.close] Closing Neo4j drivers")
        try:
            if hasattr(self, 'async_driver'):
                await self.async_driver.close()
                if self.verbose:
                    print(
                        "[Neo4jStatePersistence.close] Async driver closed successfully")
        except Exception as e:
            if self.verbose:
                print(f"[Neo4jStatePersistence.close] Error closing drivers: {e}")

    def __del__(self):
        """Ensure drivers are closed when the object is destroyed."""
        if self.verbose:
            print("[Neo4jStatePersistence.__del__] Cleaning up Neo4j drivers")
        try:
            if hasattr(self, 'async_driver'):
                # Create a new event loop for cleanup if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self.close())
        except Exception as e:
            print(f"[Neo4jStatePersistence.__del__] Error during cleanup: {e}")


    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def snapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
        """
        Snapshot the state of a graph, when the next step is to run a node.
        This method should add a NodeSnapshot to persistence.
        :param state: StateT | The state of the graph. | required
        :param next_node: BaseNode[StateT, Any, RunEndT] | The next node to run. | required
        """
        if self.verbose:
            print(
                f"[Neo4jStatePersistence.snapshot_node] Creating snapshot for node {next_node.get_node_id()}")
        await self._save(NodeSnapshot(state=state, node=next_node))

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        """
        Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.
        This method will generally call snapshot_node but should do so in an atomic way.
        :param snapshot_id: str | The ID of the snapshot to check. | required
        :param state: StateT | The state of the graph. | required
        :param next_node: BaseNode[StateT, Any, RunEndT] | The next node to run. | required
        """
        if self.verbose:
            print(
                f"[Neo4jStatePersistence.snapshot_node_if_new] Checking snapshot {snapshot_id} for node {next_node.get_node_id()}")

        async with self.async_driver.session() as session:
            query_result = await session.run(
                """
                MATCH (s:NodeSnapshot {id: $id})
                RETURN s as snapshot
                """,
                {
                    "id": snapshot_id
                }
            )

            snapshot = await query_result.single()
            if not snapshot:
                if self.verbose:
                    print(
                        f"[Neo4jStatePersistence.snapshot_node_if_new] Creating new snapshot for {snapshot_id}")

                await self._save(NodeSnapshot(state=state, node=next_node))
            else:
                if self.verbose:
                    print(
                        f"[Neo4jStatePersistence.snapshot_node_if_new] Snapshot {snapshot_id} already exists")

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        """
        Snapshot the state of a graph when the graph has ended.
        This method adds an EndSnapshot to persistence in Neo4j.

        Args:
            state: The state of the graph.
            end: Data from the end of the run.
        """
        if self.verbose:
            print("[Neo4jStatePersistence.snapshot_end] Creating end snapshot")
        await self._save(EndSnapshot(state=state, result=end))

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        """
        Record the run of the node, or error if the node is already running.
        In particular this should set:
        - NodeSnapshot.status to 'running' and NodeSnapshot.start_ts when the run starts.
        - NodeSnapshot.status to 'success' or 'error' and NodeSnapshot.duration when the run finishes.
        :param snapshot_id: str | The ID of the snapshot to record. | required
        :raises GraphNodeStatusError |  if the node status is not 'created' or 'pending'. 
        :raises LookupError  |  if the snapshot ID is not found in persistence.
        :return: AsyncContextManager[None] | An async context manager that records the run of the node.
        """


        if self.verbose:
            print(
                f"[Neo4jStatePersistence.record_run] Starting run for snapshot {snapshot_id}")
        async with self.async_driver.session() as session:
            query_result = await session.run(
                """
                MATCH (s:NodeSnapshot {id: $id})
                RETURN s as snapshot
                """,
                {
                    "id": snapshot_id
                }
            )
            single_snapshot = await query_result.single()
            if single_snapshot is None:
                raise LookupError(
                    f"No snapshot found with id={snapshot_id!r}")

            snapshot = NodeSnapshot(**single_snapshot.get('snapshot'))
            if (not snapshot):
                if self.verbose:
                    print(
                        f"[Neo4jStatePersistence.record_run] No snapshot found with id={snapshot_id}")
                raise LookupError(
                    f'No snapshot found with id={snapshot_id!r}')

            GraphNodeStatusError.check(snapshot.status)

            state_dict = json.loads(
                serialize_complex_object(snapshot.state))
            node_dict = json.loads(serialize_complex_object(snapshot.node))
            snapshot.state = self._deserialize_state(state_dict)
            snapshot.node = node_dict
            assert isinstance(
                snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
            if self.verbose:
                print(
                    f"[Neo4jStatePersistence.record_run] Setting status to running for {snapshot_id}")
            await session.run(
                """
                MATCH (s:NodeSnapshot {id: $id})
                SET s.status = 'running'
                SET s.start_ts = datetime()
                RETURN s
                """,
                {
                    "id": snapshot_id
                }
            )

        start = perf_counter()
        try:
            if self.verbose:
                print(
                    f"[Neo4jStatePersistence.record_run] Yielding control for {snapshot_id}")
            yield
        except Exception as e:
            if self.verbose:
                print(
                    f"[Neo4jStatePersistence.record_run] Error occurred for {snapshot_id}: {str(e)}")
            duration = perf_counter() - start
            await self._after_run_sync(snapshot_id, duration, 'error')
        else:
            if self.verbose:
                print(
                    f"[Neo4jStatePersistence.record_run] Success for {snapshot_id}")
            duration = perf_counter() - start
            await self._after_run_sync(snapshot_id, duration, 'success')

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        """
        Retrieve a node snapshot with status 'created' and set its status to 'pending'.
        This is used by Graph.iter_from_persistence to get the next node to run.

        Returns:
            The snapshot, or None if no snapshot with status 'created' exists.
        """
        
        if self.verbose:
            print(
                f"[Neo4jStatePersistence.load_next] Loading next snapshot for execution {self.execution_id}")
        async with self.async_driver.session() as session:
            query_results = await session.run(
                """
                MATCH (e:Execution {id: $execution_id})-[:HAS_SNAPSHOT]->(s:NodeSnapshot)
                WHERE s.status = 'created'
                ORDER BY s.start_ts DESC
                LIMIT 1
                RETURN s
                """,
                {
                    "execution_id": self.execution_id
                }
            )
            snapshot_node = await query_results.single()

            if not snapshot_node:
                print(
                    "[Neo4jStatePersistence.load_next] No snapshots found with status 'created'")
                return None

            snapshot = NodeSnapshot(**snapshot_node['s'])
            state_dict = json.loads(
                serialize_complex_object(snapshot.state))
            node_dict = json.loads(serialize_complex_object(snapshot.node))
            snapshot.state = self._deserialize_state(state_dict)
            snapshot.node = node_dict

            assert isinstance(
                snapshot, NodeSnapshot), 'Only NodeSnapshot can be loaded'
            if self.verbose:
                print(
                    f"[Neo4jStatePersistence.load_next] Setting status to pending for {snapshot.id}")

            updated_result = await session.run(
                """
                MATCH (s:NodeSnapshot {id: $id})
                SET s.status = 'pending'
                RETURN s
                """,
                {
                    "id": snapshot.id
                }
            )

            updated_snapshot_single = await updated_result.single()
            updated_snapshot = NodeSnapshot(
                **updated_snapshot_single.data()['s'])
            state_dict = json.loads(
                serialize_complex_object(updated_snapshot.state))
            node_dict = json.loads(
                serialize_complex_object(updated_snapshot.node))
            updated_snapshot.state = self._deserialize_state(state_dict)
            updated_snapshot.node = node_dict
            return updated_snapshot

    def should_set_types(self) -> bool:
        if self.verbose:
            print(
                f"[Neo4jStatePersistence.should_set_types] Checking if types need to be set: {self._state_type is None}")
        return self._state_type is None

    def set_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
        """
        Set the types of the state and run end.
        This is used to deserialize snapshots correctly.

        :param state_type: The type of the state.
        :param run_end_type: The type of the run end.
        """
        if self.verbose:
            print(
                f"[Neo4jStatePersistence.set_types] Setting types: state_type={state_type.__name__}, run_end_type={run_end_type.__name__}")
        self._state_type = state_type
        self._run_end_type = run_end_type
        self._snapshots_type_adapter = build_snapshot_list_type_adapter(
            state_type, run_end_type)

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        """
        Load the entire history of snapshots.
        load_all is not used by pydantic-graph itself, instead it's provided to make it convenient to get all snapshots from persistence.
        Returns: The list of snapshots.
        """
        if self.verbose:
            print(
                f"[Neo4jStatePersistence.load_all] Loading all snapshots for execution {self.execution_id}")
        return await self._load_async()

    async def _load_async(self) -> list[Snapshot[StateT, RunEndT]]:
        if self.verbose:
            print(
                f"[Neo4jStatePersistence._load_sync] Loading snapshots synchronously for execution {self.execution_id}")
        assert self._state_type is not None, 'State type must be set before loading snapshots'
        try:
            async with self.async_driver.session() as session:
                query_results = await session.run(
                    """
                    MATCH (e:Execution {id: $execution_id})-[:HAS_SNAPSHOT]->(s:NodeSnapshot)
                    RETURN s
                    """,
                    {
                        "execution_id": self.execution_id
                    }
                )

                query_results = await query_results.value()
                snapshots = [NodeSnapshot(**snapshot)
                             for snapshot in query_results]
                for snapshot in snapshots:
                    state_dict = json.loads(
                        serialize_complex_object(snapshot.state))
                    node_dict = json.loads(
                        serialize_complex_object(snapshot.node))
                    snapshot.state = self._deserialize_state(state_dict)
                    snapshot.node = node_dict

                return snapshots

        except FileNotFoundError:
            if self.verbose:
                print(f"[Neo4jStatePersistence._load_sync] No snapshots found")
            return []

    async def _after_run_sync(self, snapshot_id: str, duration: float, status: SnapshotStatus) -> None:
        """
        Update the status and duration of a snapshot in Neo4j.
        :param snapshot_id: str | The ID of the snapshot to update. | required
        :param duration: float | The duration of the run. | required
        :param status: SnapshotStatus | The status of the snapshot. | required
        """
        if self.verbose:
            print(
                f"[Neo4jStatePersistence._after_run_sync] Updating snapshot {snapshot_id} with duration={duration}, status={status}")

        async with self.async_driver.session() as session:
            await session.run(
                """
                    MATCH (s:NodeSnapshot {id: $id})
                    SET s.status = $status
                    SET s.duration = $duration
                    RETURN s as snapshot
                """,
                {
                    "id": snapshot_id,
                    "status": status,
                    "duration": duration
                }
            )

    async def _save(self, snapshot: Snapshot[StateT, RunEndT]) -> None:
        """
        Save a snapshot to Neo4j.
        :param snapshot: Snapshot[StateT, RunEndT] | The snapshot to save. | required
        """
        if self.verbose:
            print(
                f"[Neo4jStatePersistence._save] Saving snapshot {snapshot.id} asynchronously")
        assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'

        # Serialize state and node with complex object handling
        state_dict = serialize_complex_object(snapshot.state)

        state_json = json.dumps(state_dict)

        async with self.async_driver.session() as session:
            node_dict = serialize_complex_object(snapshot.node)
            node_json = json.dumps(node_dict)

            record = None

            if isinstance(snapshot, NodeSnapshot):
                result = await session.run(
                    """
                    MERGE (e:Execution {id: $execution_id})
                    ON CREATE SET e.created_at = datetime()
                    WITH e
                    CREATE (s:NodeSnapshot {
                        id: $id,
                        duration: $duration,
                        start_ts: $start_ts,
                        status: $status,
                        kind: $kind
                    })
                    SET s.state = $state
                    SET s.node = $node
                    WITH e, s
                    CREATE (e)-[:HAS_SNAPSHOT]->(s)
                    RETURN s as snapshot
                    """,
                    {
                        "id": snapshot.id,
                        "state": state_json,
                        "node": node_json,
                        "execution_id": self.execution_id,
                        "duration": snapshot.duration,
                        "start_ts": snapshot.start_ts,
                        "status": snapshot.status,
                        "kind": snapshot.kind
                    }
                )
                record = await result.single()
            elif isinstance(snapshot, EndSnapshot):
                result_dict = serialize_complex_object(snapshot.result)
                result_json = json.dumps(result_dict)
                result = await session.run(
                    """
                    MERGE (e:Execution {id: $execution_id})
                    ON CREATE SET e.created_at = datetime()
                    WITH e
                    CREATE (s:EndSnapshot {
                        id: $id,
                        ts: $ts,
                        kind: $kind
                    })
                    SET s.result = $result
                    SET s.state = $state
                    RETURN s as snapshot
                    """,
                    {
                        "execution_id": self.execution_id,
                        "id": snapshot.id,
                        "ts": snapshot.ts,
                        "kind": snapshot.kind,
                        "result": result_json,
                        "state": state_json,
                    }
                )
                record = await result.single()
            # Properly handle and print the result
            if self.verbose:
                if record:
                    print(
                        f"[Neo4jStatePersistence._save] Successfully saved snapshot with ID: {record['snapshot']['id']}")
                else:
                    print(
                        "[Neo4jStatePersistence._save] No record was returned from the save operation")

    def _deserialize_parts(self, parts_list: list[dict]) -> list[Union[ModelRequestPart, ModelResponsePart]]:
        """
        Helper method to reconstruct message parts from dictionaries
        :param parts_list: list[dict] | The list of message parts to deserialize. | required
        :return: list[Union[ModelRequestPart, ModelResponsePart]] | The deserialized message parts.
        """

        deserialized_parts = []
        for part in parts_list:
            part_kind = part.get('part_kind', '')
            if part_kind == 'user-prompt':
                deserialized_parts.append(UserPromptPart(**part))
            elif part_kind == 'system-prompt':
                deserialized_parts.append(SystemPromptPart(**part))
            elif part_kind == 'text':
                deserialized_parts.append(TextPart(**part))
            elif part_kind == 'tool-call':
                deserialized_parts.append(ToolCallPart(**part))
            elif part_kind == 'tool-return':
                deserialized_parts.append(ToolReturnPart(**part))
            elif part_kind == 'retry-prompt':
                deserialized_parts.append(RetryPromptPart(**part))
            else:
                # If we don't recognize the part_kind, keep it as a dict
                deserialized_parts.append(part)
        return deserialized_parts

    def _deserialize_messages(self, messages_list: list[dict]) -> list[ModelMessage]:
        """
        Helper method to reconstruct ModelMessages from dictionaries
        :param messages_list: list[dict] | The list of message parts to deserialize. | required
        :return: list[ModelMessage] | The deserialized message parts.
        """

        deserialized_messages = []
        for msg in messages_list:
            # Process parts first
            if 'parts' in msg and isinstance(msg['parts'], list):
                msg['parts'] = self._deserialize_parts(msg['parts'])

            # Create the appropriate message type
            if msg.get('kind') == 'request':
                deserialized_messages.append(ModelRequest(**msg))
            elif msg.get('kind') == 'response':
                deserialized_messages.append(ModelResponse(**msg))
            else:
                # Keep it as a dict if we can't determine the type
                deserialized_messages.append(msg)
        return deserialized_messages

    def _deserialize_state(self, state_dict: dict) -> StateT:
        """
        Helper method to reconstruct the state with proper object types
        :param state_dict: dict | The state to deserialize. | required
        :return: StateT | The deserialized state.
        """

        # Handle the message lists in the state
        if 'ask_agent_messages' in state_dict and isinstance(state_dict['ask_agent_messages'], list):
            state_dict['ask_agent_messages'] = self._deserialize_messages(
                state_dict['ask_agent_messages'])

        if 'evaluate_agent_messages' in state_dict and isinstance(state_dict['evaluate_agent_messages'], list):
            state_dict['evaluate_agent_messages'] = self._deserialize_messages(
                state_dict['evaluate_agent_messages'])

        # Create the state object
        return self._state_type(**state_dict)
