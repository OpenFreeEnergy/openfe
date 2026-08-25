# CLI: Worker-based Execution 

This tutorial demonstrates how to setup, run, and gather using **openfe**'s worker-based CLI execution.

## Setting up a Campaign

``` bash
> openfe plan-rbfe-network --warehouse
```

OR

```bash
> openfe setup-warehouse-db --alchemical-network tyk2_alchemical_network.json
```

## Running the Campaign

To execute a single unit, run:

``` bash
> openfe run-worker tyk2.db tyk2.db
```

In practice, you will likely be submitting many workers simultaneously using SLURM or similar. 

... insert run_worker.sh

## Gathering Results


```bash
> openfe output-as-legacy-json warehouse
```


