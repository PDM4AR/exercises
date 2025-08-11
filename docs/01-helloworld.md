---
layout: default
---

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

- Run `git clone <>` where `<>` is either HTTPS or SSH depending on how you set up git. 
- Navigate to the `exercise` folder, where the exercise repo is cloned locally. Open VS Code in the repository by running `code .` in the terminal.
- Alternatively, open VS code then click select File -> Open and select *the entire folder*.
- VS Code will propose to install "Dev Container". Click "install".
- VS Code will give you a message similar to:

> Folder contains a Dev Container configuration file. Reopen folder to develop in a container.

- Select "Reopen in container". Now you should have the folder open while VS Code is in "container development mode".
- Alternatively, you can press `ctrl+shift+P` and then search for the command "Reopen in container".
- Create a new terminal using Terminal -> New Terminal.
- Now browse the Python files in `src/`. Verify that autocompletion works.

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
Make sure that you opened the project in the container without errors.

You can run it using the preinstalled hooks in `.vscode` folder:
Click on the Run & Debug icon and select from the dropdown menu (*Run* mode ignores breakpoints and debug settings, while *Debug* mode stops at breakpoints):

![image](https://user-images.githubusercontent.com/18750753/194089273-dc9f95e4-0553-45c4-a261-233727ef72ae.png)

You can also run it from the VS Code terminal in a equivalent way:
```bash
python3 [path/to/]src/pdm4ar/main.py --exercise [exercise ID]
### # e.g. to run the first exercise (no debug)
### python3 src/pdm4ar/main.py --exercise "01"
```

You should then find in the `out/` folder a _html_ report that gets generated.
You can open it from the filesystem in you favorite browser or simply right click on the html file and ``Open with Live Server''
Here is an example for the lexi comparison:
![image](https://user-images.githubusercontent.com/18750753/194091460-4e0896ea-26fa-4f43-a4b2-341991da0e5a.png)

### Creating local test cases

You might have noticed that the expected outputs in the above report are always `None` instead of the true answers. This is apparently not very helpful for debugging your algorithm. How can you fix this? Take a look at the function `get_exercise1()` in the file `src/pdm4ar/exercises_def/ex01/ex01.py`. In this function, an `Exercise` object is constructed with a description(`desc`), the evaluation functions(`evaluation_fun` and `perf_aggregator`), the test cases(`test_values`) and the expected results(`expected_results`). The last two arguments are relevant for creating the local test cases. Notice that in line 71, the expected results are declared as a list of `None`. This is why you see it in the report. Try to play around with it and observe the change in the generated report.

In all the following exercises, we will provide you with some local test cases and the true answers to them. Nonetheless, feel free to create your own test cases in the same function of other exercises(e.g.`get_exercise2()` in `src/pdm4ar/exercises_def/ex02/ex02.py`). 

### Submitting your solution to the server

Once you are happy with your solution, you can submit it to the server for evaluation.
To do so, it is sufficient to *push* a commit with special keywords in the commit message.
The keywords are:
```[submit][xx] your commit message```
where `xx` is the exercise ID you want to submit for evaluation.

For example, if you want to submit your solution for exercise 1 (after implementing a solution):
commit and push your changes with the following message:
```[submit][01] luky solution```
After a few minutes you should see your summary results directly on the Github pull request named Feedback.

**Important**: 
- Your grade for the upcoming exercises will depend *only* on the ***last valid*** submission that you make for that specific exercise.
Make sure to extensively test locally your solution before submitting it.
- Be aware that due to the large number of submissions before the deadline, it typically takes longer for you to receive the evaluation outcome from the server. A submission is valid as long as you received the message titled `Submission queued` on the Github pull request.