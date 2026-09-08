EITN Workspace
==============

This folder is shared between your computer and the Docker container.
Files you save here are accessible from both sides and persist across
container restarts and rebuilds.

Inside the container this folder is mounted at: /workspace

Virtual environments
--------------------
  use-nrn   NEURON 8.2.6 (default)
  use-bsb   BSB + NEST 3.8 + cerebellar-models
  use-mfm   BSB + NEST 3.8 + TVB + JupyterLab + cerebellar-models

Quick start
-----------
  docker compose up -d                  Start the container
  docker exec -it eitn-devenv bash      Open a shell
  use-mfm                               Switch to the mfm environment
  jupyter lab --ip=0.0.0.0 --port=8080 --no-browser   Start JupyterLab
