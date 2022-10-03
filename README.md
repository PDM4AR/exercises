# PDM4AR-exercises

Website: [https://pdm4ar.github.io/exercises/](https://pdm4ar.github.io/exercises/)

## gen report locally

### The easy way
look at the vscode side panel, click on the "Run and Debug" icon. On the top, select the exercise, in debug mode or run mode. Click the triangle "Play" button next to the selection.

### the way used in evaluation server
```bash
# in each new vscode terminal
pip3 install -e .

python3 [path/to/]src/pdm4ar/main.py --exercise [name of exercise]
# e.g.
python3 /workspaces/exercise1-student/src/pdm4ar/main.py --exercise exercise1
```

### Kept for ref: with compmake
```bash
# in each new vscode terminal
pip3 install -e .

# generate report in local_reports/$ex
ex=exercise1 && pdm4ar-exercise -o local_reports/$ex --reset -c "rparmake" --exercises $ex
```

