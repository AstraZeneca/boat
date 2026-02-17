Generate a component
--------------------------------
## Important Action Items

In `pyproject.toml`, modify `name` and `description`.

In `Dockerfile`, modify the `ENTRYPOINT`.

In `~.github/workflows/build-images.yaml`, add your component to the job list by following the `build-component-1` example replacing the inputs:
* project-name: this is the harbor project where the image will be stored
* image-name: this is the name of the image that will be pushed to harbor.csis.astrazeneca.net/{project-name}
* src-path: the path to your component, relative to the repo's root

Reach out to the CAI engineering team for help with setting up the final stable image building.
