# Hello world :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td>
  </tr>
</table>


The exercises will be carried out with the help of Github classrooms.
We give you a link for a "homework". 
By accepting it, you will create you personal private repository from the exercise template.

## Create your repository

On Piazza you will find an invitation link to the Github Classroom exercise.
With a few clicks  you will associate your Github account to your _Legi_ number and create your own private repository.

Once your repository is created, clone it on your local computer and open it in VS Code:

- Select File -> Open and select *the entire folder*.
- VS Code will propose to install "Dev Container". Click "install".
- VS Code will give you a message similar to:

> Folder contains a Dev Container configuration file. Reopen folder to develop in a container.

- Select "Reopen in container". Now you should have the folder open while VS Code is in "container development mode".
- Create a new terminal using Terminal -> New Terminal.
- Now browse the Python files in `src/`. Verify that autocompletion works..


#### Setting up the _git remote_ repository for updates

From time to time we might release fixes and/or new exercises on the "parent" repository
(the template from which your repo was created).
In order to get the updates in the future we need to tell git that there exist a _template_ repository to pull from.
You can set this running `make set-template`.
Verify that everything went well typing `git remote -v` in your terminal:

* origin should point to your repository;
* template should point to the template repository.

Then update via `make update`, if your OS does not support Makefiles, 
you can run the commands from the terminal directly by copying them from the Makefile.
It could be that you have to commit the merge of the incoming updates from the template repository.

# Exercise1 - Lexicographic comparison [Not graded]

Let's now try to solve our first exercise! It will be a simple one.

First, open the source folder (`src`) and have a look at the structure:

The exercises that you need to modify are inside the `exercises` folder. 
Do **not modify** any other file which is not inside `exercises`, it might result in breaking everything.
Now open the file `exercises/ex01/ex1.py` and try to implement a function that compares two vectors according to a lexicographic order. 
For each entry, lower is better.

Note that the function is _type annotated_:
![Screenshot from 2021-09-21 10-12-57](https://user-images.githubusercontent.com/18750753/134135930-884af68d-f5d9-4a00-b06f-f911468c400b.png)
Despite Python is not a strongly typed language as C++ or similar, python annotations are a great way to develop with
clear interfaces. Learn more [here](https://www.python.org/dev/peps/pep-0484/).

Something like this

```python
def compare(a: Tuple[float], b: Tuple[float]) -> ComparisonOutcome:
```

reads as follows:
> _"compare is a function that takes two arguments, each is expected to be a tuple of floats. The type of the returned argument should be ComparisonOutcome"_

### Evaluating locally your solution

You are now ready to implement your solution and check it locally.
You can run it using the preinstalled hooks in `.vscode` folder.
First, install the pdm4ar as a python package running in the terminal: ``
```bash
pip3 install -e [path/to/exercises_repo]
### e.g.:
### pip3 install -e .
```
Then, click on the Run & Debug icon and select from the dropdown menu (*Run* mode ignores breakpoints and debug settings, while *Debug* mode stops at breakpoints):

![image](https://user-images.githubusercontent.com/18750753/194089273-dc9f95e4-0553-45c4-a261-233727ef72ae.png)

You can also run it from the VS Code terminal in a equivalent way:
```bash
pip3 install -e [path/to/exercises_repo]
python3 [path/to/]src/pdm4ar/main.py --exercise [exercise ID]
### # e.g. to run the first exercise (no debug)
### pip3 install -e .
### python3 src/pdm4ar/main.py --exercise "01"
```

You should the find in the `out/` folder a _html_ report that gets generated.
You can open it from the filesystem in you favorite browser!
Here is an example for the lexi comparison:
![image](https://user-images.githubusercontent.com/18750753/194091460-4e0896ea-26fa-4f43-a4b2-341991da0e5a.png)



### Submitting your solution to the server

Once you are happy with your solution, you can submit it to the server for evaluation.
To do so, it is sufficient to *push* a commit with special keywords in the commit message.
The keywords are:
```[submit][xx] your commit message```
where `xx` is the exercise ID you want to submit for evaluation.

For example, if you want to submit your solution for exercise 1 (after implementing a solution):
commit and push your changes with the following message:
```[submit][01] luky solution?```
After a few minutes you should see your summary results directly on the Github pull request named Feedback.

**Important**: your grade for the upcoming exercises will depend *only* on the ***last valid*** submission that you make for that specific exercise.
Make sure to extensively test locally your solution before submitting it.
