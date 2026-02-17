![Python version](https://img.shields.io/badge/python-3.8-blue.svg)

# cai-boat

--------
## Important

Please always create a separate environment for your project. There is a handy utility provided that could do
this for you, [read below](#make-init).

Please do not put confidential data useful for this project anywhere other than data folder.
The .gitignore is designed to ignore the folder contents making the project git safe.

Please keep your pyproject.toml and poetry.lock up to date with environments dependencies.
You can find more information about the [documentation and set up here](https://python-poetry.org/docs/).

If your project version supports a different python version, please change in the above badges along with
the pyproject.toml and Dockerfile.

Please remember to edit the pyproject.toml to suit your project. We are hoping to automate it as GitHub templates
allow for it in the future.


--------
## Actions

We are actively developing and maintaining Github Actions specifically for internal use in AstraZeneca.
 You can find the list of actions by
 [searching in azu-ignite org for repositories](https://github.com/azu-ignite?q=az-action&type=&language=&sort=)
 starting with az-action-.

**Composite workflows available in this template:**

1. [Publish Package](https://github.com/azu-biopharmaceuticals-rd/data-science-template/blob/main/.github/publish-package/action.yaml) - change/set in the inputs the JFrog Artifactory user and repository your project will use.
2. [Pytest Feature](https://github.com/azu-biopharmaceuticals-rd/data-science-template/blob/main/.github/pytest-feature/action.yaml) - change/set in the inputs the JFrog Artifactory user for your project.
3. [Build Component](https://github.com/azu-biopharmaceuticals-rd/data-science-template/blob/main/.github/workflows/build-component.yaml) - change/set in the inputs the Harbor project where your docker images will be stored. This workflow can be triggered manually from the [GitHub Actions](https://github.com/azu-biopharmaceuticals-rd/cai-boat/actions/workflows/build-component.yaml) tab.

  > Make sure the CI secrets match your project's secrets!

**Actions that run on this project by default:**

1. PR checks
    - Linting, Testing and Semantic Versioning happen when a PR is opened. In adition to this, SonarQube checks are available for your project upon request. Make sure you have the required credentials/secrets to run those.
2. Version & Release; Publish
    - Use Semantic Versioning for your project. This runs on push to main and relies on conventional commits, there is a check for this in the pre-commit hooks.
    - After the version has been updated and released in GitHub, the package can be published to Artifactory. Uncomment this step if you want the package to be released.
3. Package Manual Release
    - For pre-release publishing of your package, trigger this workflow manually from the [GitHub Actions](https://github.com/azu-biopharmaceuticals-rd/cai-boat/actions/workflows/manual-pre-release.yaml) tab.
4. Docker images build and publish
    - Use `build-images.yaml` to build each individual component you need. Check the inputs, and change the secrets to match the secrets your repo has access to.
5. Build Documentation
    - Build documentation for the project, ready for use with github pages

--------
## Project Organization


    ├── .github/workflows  <- Github actions.
    ├── Makefile           <- Makefile with utility commands.
    ├── Dockerfile         <- Create a Docker image for this project.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- The data folder for this project. This is always ignored by git by design.
    │  
    ├── docs               <- A default mkdocs project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Please ensure the output is stripped out of any
    confidential information.
    │  
    ├── Dockerfile         <- A generic Dockerfile to help you get started.
    ├── cai_boat                <- Source code for use in this project.
    │

--------
## Set up Documentation Web Page

You can get your project documentation as a github website out of the box using this template.
To enable it you must:

1. Go to **Settings**.
1. Under the **Code and automation** header in the sidebar click **Pages**.
1. Under **Build and deployment** select **Deploy from a branch**
1. Choose **Branch** select **gh-pages**
1. Scroll to the bottom of the page and go to the url in the **Enforce HTTPS** section.

You should be able to view your documentation site! You can change the colours and some of the formatting in the `mkdocs.yaml` file.

--------
## Utilities Available:

### make init
Would create a pipenv environment for you along with pre-commit

### make clean
A utility tool to clear pycache and pyc or pyo files

If you find any issues or feature requests with the template, please [raise it here](https://github.com/azu-ignite/data-science-template)
