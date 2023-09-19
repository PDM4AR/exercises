python_version=3.11-bullseye
registry=idscfrazz
image_name=pdm4ar2023
dockerfile=Dockerfile.base

platforms=linux/amd64,linux/arm64
# platforms=linux/arm64
tag=$(registry)/$(image_name):$(python_version)


build:
	docker buildx build --platform $(platforms) -f $(dockerfile) -t $(tag) --build-arg VARIANT=${python_version} --push .

build-no-cache:
	docker buildx build --platform $(platforms) -f $(dockerfile) -t $(tag) --build-arg VARIANT=${python_version} --push . --no-cache

push-on-dockerhub:
	docker login
	docker push $(tag)
