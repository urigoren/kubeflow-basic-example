import os
from pathlib import Path
from subprocess import call
component_path = Path("dockers")
for p in component_path.iterdir():
    cmd = """
        docker build -t {component} dockers/{component}
        docker tag preprocess gcr.io/kubeflow-demos/{component}:latest
        docker push gcr.io/kubeflow-demos/{component}:latest
    """.format(component=p.name)
    call(cmd,shell=True)
