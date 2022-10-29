# Preliminaries

- [Rules of the game](#rules-of-the-game)
- [Getting started](#getting-started)
    * [Git](#git)
        - [Installing Git](#installing-git)
        - [Creating a Github account](#creating-a-github-account)
        - [Resources](#-resources--)
        - [Goals](#--goals---)
    * [Docker](#docker)
        - [Resources](#-resources---1)
        - [Goals](#--goals----1)
    * [Python](#python)
        - [Resources](#-resources---2)
    * [IDEs (optional-recommended)](#ides---optional---recommended----)
- [Support](#support)

## Rules of the game

The programming exercises are a great way to understand and apply what you learn in class.

There will be three types of exercises:

- **Warm-up** exercises, not graded;
- **Weekly** exercises, graded (will account for 30% of the final grade);
- **Mid/Final term** exercises, graded (will account for 70% (30 + 40) of the final grade).

The *not-graded* ones will be a warm-up, making sure everything works as expected, gives time to catch up with python if
you ar not familiar with it.

#### **Weekly** exercises, graded.
The weekly exercises will be released almost on a weekly basis.
They will involve relative short programming task related to the topics covered in class.
You will submit your solutions via your private repository on Github (detail instructions on the submission procedure can be found later on).


**Important information**:

* You will get a score 0-100 for each exercise. A missing submission (or failing to meet the deadline) will count as 0. The score will be based on some performance metrics (how correct is your solution, how fast it is, how much memory
it uses, etc.). Performance criteria will be declared in the exercise description.
* Only the best _N-1_ submissions will account for the 30% of your final grade, where _N_ is the number of exercises.
* Only the last submission within the exercise time window will account for your exercise grade. 
* We limit the number of submissions per day to *1* per person per exercise.
Make sure to extensively test your code locally, consider this as a learning experience for robotics, field tests are expensive.
* Here you can see a temporary schedule (will be completed on the fly):

| Exercise **ID** | **Topic**             | Evaluation **opens** | Evaluation **closes** | **Deadline status** |
|-----------------|-----------------------|----------------------|-----------------------|---------------------|
| 02              | Graph Search          | Oct. 8th             | Oct. 26th (23:59 CET) | CONFIRMED (EXTENDED)|
| 03              | Informed Graph Search | Oct. 13th            | Oct. 26th (23:59 CET) | CONFIRMED (EXTENDED)|
| 04              | Dynamic Programming   | Oct. 19th            | Nov. 4th  (23:59 CET) | CONFIRMED (EXTENDED)|
| 05              | Steering (Dubins)     | Nov. 1st             | Nov. 15th (23:59 CET) | TENTATIVE           |
| 06              | Optimization          | TBD                  | TBD (23:59 CET)       | TBD                 |
| 07              | Collision Checking    | TBD                  | TBD (23:59 CET)       | TBD                 |

* Most of these exercises are already accessible before the evaluation opens. 
You are free to read them but be aware that we might still decide to change them until the evaluation opens.
After the evaluation opens the exercise is "frozen" and you can solve it and submit it.

#### **Mid/Final term** exercises, graded.
Same modality as the weekly exercises, but they will be a bit more involved.

| Exercise **ID** | **Topic** | Evaluation **opens** | Evaluation **closes** |
|-----------------|-----------|----------------------|-----------------------|
| TBD             | Surprise  | Nov. 7th (TBC)       | Nov. 31st (TBC)       |
| TBD             | Surprise  | Dec. 15th            | Jan. 15th (23:59 CET) |


## Plagiarism and dishonest conduct

We take plagiarism and dishonest conduct very seriously.
The exercises are meant to be solved **individually**. 
We will check for plagiarism with automated software and human help. 
Violating the [ETH plagiarism etiquette](https://ethz.ch/content/dam/ethz/main/education/rechtliches-abschluesse/leistungskontrollen/plagiarism-citationetiquette.pdf)
will result in consequences according to the ETH regulations:

>"C. The consequences of plagiarism Pursuant to Art. 2 Para. b of the ETH Zurich Disciplinary Code (RSETHZ 361.1) plagiarism constitutes a disciplinary violation and will result in disciplinary procedures.
> Detailed information regarding these procedures and their jurisdiction may be found in the ETH Zurich Disciplinary Code (RSETHZ 361.1 / www.rechtssammlung.ethz.ch)."


## Getting started

We will use:

- [Git](https://git-scm.com/) as version control system;
- [Python](https://www.python.org/) as programming language;
- [Docker](https://www.docker.com/) as environment containerization (but you won't see it);

If they all sound completely new to you do not panic. 
We will require a very basic use of most of them, but it is a good time to start learning these tools since they are all widely adopted in modern robotics.
If you get stuck in the process try to pair up with some more experienced colleagues who can help you. 
If this still does not solve the problem, try to reach out to the instructors on Piazza. 
If you are using Mac or Linux-based OS the process should be straight forward. 
Windows can give some more hiccups, but it is supported as well.

### Git

Git is a version control software.
Please find more explanation under the "Resources" paragraph if you have never heard of it before. 
You will need Git on your computer and a GitHub account.

#### Installing Git

Simply follow the steps for your OS at [this link](https://git-scm.com/downloads)

### Creating a GitHub account

If you do not have already a GitHub account create a new one [here](https://github.com/join)

#### _Resources_

- [Git introduction](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/preliminaries_git.html#sec:preliminaries-git)
- [Github tutorial](https://guides.github.com/activities/hello-world/)

#### **Checklist**

- [ ]  I have Git set up on my computer
- [ ]  I have a GitHub account
- [ ]  I know how to clone a repository
- [ ]  I have cloned the GitHub repository

### Docker

We will run the exercises in a virtual environment (or better, in a container).
Read the "Resources" section to get a better understanding of it, containerization is ubiquitous in modern software
development.
Now let's install it on your computer:

* (Mac, Linux) [installation instructions](https://docs.docker.com/get-docker/)
* (Windows) the procedure is more complicated:
    + Follow the manual installation steps for Windows Subsystem for
      Linux [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10). On step 1, follow the recommendation of
      updating to WSL 2. On step 6 you can download Ubuntu 18.04 LTS. You do not necessarily need to install Windows
      Terminal.
    + Now go [here](https://docs.docker.com/desktop/windows/install/) and follow the _Install Docker Desktop on Windows_
      instructions. You can then start Docker Desktop and follow the *quick start guide*.

#### _Resources_

- [Docker intro](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/preliminaries_docker_basics.html)

#### **Checklist**

- [ ] I have installed Docker on my computer
- [ ] I can run without errors the Docker hello-world (`docker run hello-world`)


### Python

Python will be the programming language adopted in this course.

#### _Resources_

- [Official documentation](https://docs.python.org/3/)
- [Python Tutorial](https://www.tutorialspoint.com/python/index.htm), in particular make sure to go through the basic
  data structures (tuples, lists, dictionaries, sets,...), loops (while, for,...), conditional statements (if-else),
  functions, classes and objects.

### VS Code

Using an [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) is not necessary.
But it provides a good set of tools that speed up the development (code navigation, debugging,...).
Moreover, we will provide environment configurations that will make your life easier.

Install [VS Code](https://code.visualstudio.com/)

There are many other good IDEs for python (PyCharm, Atom, Spyder,...), they should work just fine if you know how to
replicate exactly the development evironment in `.devcontainer/` but we won't support them officially.


## Support

Use the forums on Piazza for general questions: this way, you can help other students who experience the same issues.
