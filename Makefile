##### to get the new exercises run "make update"
set-upstream: # Run this only once to set the upstream
	git remote add template git@github.com:PDM4AR/exercises.git

update: # todo check if this works
	git pull template master --allow-unrelated-histories
