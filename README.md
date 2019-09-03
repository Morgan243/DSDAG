# DSDAG - The Data Science DAG ![DAG](https://i.stack.imgur.com/bfnEB.png)

**This is a work in progress**

![Build Status](https://travis-ci.com/Morgan243/DSDAG.svg?token=AKmRHsSyxe3XyR58fo7N&branch=attrsdev)
### Goals
- Capture data processing knowledge in a way that's self-documenting and reusable
- Reduce the time to implement complex analysis by reusing related prior work
- Facilitate easier debugging and arbitrary partial execution of complex projects

#### Why not Luigi, Airflow, ... etc?

- DSDAG targets **interactive environments** (jupyter)
    - Other tools focus on integration ETL & data engineering systems
- DSDAG requires **less boilerplate** code (but is very similar to Luigi in structure)
- DSDAG can cache (in memory or on disk) and provide access to **Intermediate outputs**


## Approach
- The *DSDAG* framework implements several core components:
    - *Ops* (or *Processes*): The smallest unit of compute, similar to a function
    - *Parameters*: User defined arguments to *processes* that are tracked
    - *DAG*: A collection of processes that are needed to compute an output
- Users define Operations that process inputs or provide some output
- Operations can specify other operations that they depend on, but these inputs can be overridden at runtime
- Runtime decisions or parameters (e.g. time of day, policy number, etc.) are implemented as *Parameters* to the *Process*
- Outputs are materialized by building a DAG and running it