# Hello world :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td>
  </tr>
</table>

In this tutorial we will test that you have all the tools set up correctly.

### Create your repository from the template

git clone the repository then:

- Select File -> Open and select *the entire folder*.
- VS Code will propose to install "Dev Container". Click "install".
- VS Code will give you a message similar to:

> Folder contains a Dev Container configuration file. Reopen folder to develop in a container.
> Select "Reopen in container".

- Now you should have the folder open while VS Code is in "container development mode".
- Create a new terminal using Terminal -> New Terminal.
- TODO the testing part
- Now browse the Python files in `src/`. Verify that autocompletion works..



#### Setting up the upstream branch for updates

From time to time we might release updates or new exercises on the "parent" repository (the one you forked from).
In order to simply get the updates in the future we need to tell git that there exist an _upstream_ repository to pull from.
You can set this running `make set-upstream`.

[todo update] 

Verify that everything went well typing `git remote -v` in your terminal. You should get something like this (`alezana` -> `YOUR GITHUB USERNAME`):
![image](https://user-images.githubusercontent.com/18750753/137486162-4aa48862-0c52-4e7d-987f-6d4b57582ad1.png)

# Exercise1 - Lexicographic comparison

Let's now try to solve our first exercise! It will be a simple one.

First, open the source folder (`src`) and have a look a the structure:

The exercises that you need to modify are inside the `exercises` folder. Do **not modify** any other file which is not
inside `exercises`, it might result in breaking everything. Now open the file `exercises/ex01/ex1.py` and try to
implement a function that compares two vectors according to a lexicographic order. For each entry, lower is better.

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

You are now ready to implement your solution and check it with

[todo]

Open the generated report and check your results!
