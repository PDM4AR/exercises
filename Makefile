##### to get the new exercises run set-template (needed only once) and then "make update"
set-template:
	git remote add template git@github.com:PDM4AR/exercises.git

update:
	git pull -X theirs template master --allow-unrelated-histories

# Command to update the base image on top of which the devcontainer is built.
# Note that this needs to be run from outside the devcontainer.
CURRENT_BASE = pdm4ar2024:3.11-bullseye
update-base-image:
	docker pull idscfrazz/$(CURRENT_BASE)
