python_version=3.9-bullseye
registry=idscfrazz
image_name=pdm4ar2022
dockerfile=Dockerfile.base

tag=$(registry)/$(image_name):$(python_version)


build:
	docker build -f $(dockerfile) -t $(tag) . --build-arg VARIANT=${python_version}

build-no-cache:
	docker build --no-cache -f $(dockerfile) -t $(tag) . --build-arg VARIANT=${python_version}

push-on-dockerhub:
	docker login
	docker push $(tag)
