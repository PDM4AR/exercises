### Instructions ###
#
#To get the new exercises run set-template (needed only once) to set the template repository as a remote
#and then "make update" every time you want to update the exercises.
# Note that since your personal repository and the template have not a common git history,
# this is a dangerous operation that might result in overwrite of local changes and/or conflicts.
set-template:
	git remote add template git@github.com:PDM4AR/exercises.git

update:
	git pull -X theirs template master --allow-unrelated-histories

# Command to update the base image on top of which the devcontainer is built.
# Note that this needs to be run from outside the devcontainer.
CURRENT_BASE = pdm4ar2024:3.11-bullseye
update-base-image:
	docker pull idscfrazz/$(CURRENT_BASE)
