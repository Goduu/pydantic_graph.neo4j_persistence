
# Pydantic Graph - Neo4j Persistence

[![PyPI Version](https://badge.fury.io/py/pydantic-graph-neo4j-persistence.svg)](https://pypi.org/project/pydantic-graph-neo4j-persistence/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

`pydantic_graph_neo4j_persistence` is a persistence adapter for Neo4j that seamlessly integrates with Pydantic models. It enables saving node snapshots directly into a Neo4j database, facilitating efficient graph data management within Python applications.

## Features

- **Seamless Integration**: Effortlessly combine Pydantic's data validation with Neo4j's graph database capabilities.
- **Automatic Snapshotting**: Automatically persist node states to Neo4j, ensuring data consistency.
- **Flexible Configuration**: Tailor the persistence layer to fit various application architectures.

## Installation

To install `pydantic_graph_neo4j_persistence`, use pip:

```bash
pip install pydantic_graph_neo4j_persistence
```
# Quick Start

Here is a basic example demonstrating how to use pydantic_graph.neo4j_persistence:

```python
from pydantic_graph_neo4j_persistence import Neo4jStatePersistence

async def run_as_cli(answer: str | None):
    persistence = Neo4jStatePersistence(
        uri="your_uri",
        username="your_username",
        password="your_password",
        execution_id="execution_id",
        verbose=False
    )

    persistence.set_graph_types(my_pydantic_graph)

    ...
    


if __name__ == '__main__':
    import asyncio
    import sys
    
    a = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(run_as_cli(a))
```

You can see a full example on the oficial pydantic [documentation](https://ai.pydantic.dev/examples/question-graph/). From there just change the persistence variable to use the `Neo4jStatePersistence`

# Nodes and Relationships
The Neo4jStatePersistence uses and require the following Nodes: 
- NodeSnapshot
- EndSnapshot
- Execution

And the following relationship:
- HAS_SNAPSHOT

(e:Execution) ->[:HAS_SNAPSHOT]->(n:NodeSnapshot)

(e:Execution) ->[:HAS_SNAPSHOT]->(n:EndSnapshot)


# Contributing
We welcome contributions to enhance the functionality and usability of 
pydantic_graph.neo4j_persistence. To contribute:

    Fork the repository.

    Create a new branch for your feature or bug fix.

    Ensure existing tests pass and add new tests for your changes.

    Submit a pull request detailing your changes and the problem they address.

# License

pydantic_graph.neo4j_persistence is licensed under the MIT License. See the LICENSE file for details.
