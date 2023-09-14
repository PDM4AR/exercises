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

There will be four types of exercises:

- **Warm-up** exercises, not graded;
- **Weekly** individual exercises, graded (will account for 30% of the final grade);
- **Finals** group exercises, graded (will account for 70% (30 + 40) of the final grade).
- **Extra** exercises from previous years, not graded.

#### **Warm-up** exercises, not graded.

Just ensuring everything is working as expected.

#### **Weekly** individual exercises, graded.

The weekly exercises must be solved *individually*.
They will involve relative short programming task related to the topics covered in class.
You will submit your solutions via your private repository on Github
(detailed instruction on the submission procedure can be found later on).

**Important information**:

* Each exercise has a limited submission window, typically 2 weeks.
* A missing submission (or failing to meet the deadline) will count as 0.
  The score will be based on some performance metrics (how correct is your solution, how fast it is, how much memory
  it uses, etc.) which will be declared in the exercise description.
  Note that we will **not** provide in advance the mapping _score_ -> _grade_.
* Only the last **valid** (i.e., processed by the server irrespectively of the outcome) submission within the exercise
  time window will account for your exercise grade.
* Only your best _N-1_ exercise results contribute to the 30% of your final grade.
  Where _N_ is the total number of weekly exercises.
* For each exercise you will have a limited number of possible submissions to the server. You will be able to test
  locally as much as you want though.
  Consider this as a learning experience for robotics, field tests are expensive.
* Here you can see a temporary schedule (will be completed on the fly):

| Exercise **ID** | **Topic**             | Evaluation **opens** | Evaluation **closes** | **Deadline status** |
|-----------------|-----------------------|----------------------|-----------------------|---------------------|
| 02              | Graph Search          | -                    | - (23:59 CET)         | TBD                 |
| 03              | Informed Graph Search | -                    | - (23:59 CET)         | TBD                 |
| 04              | Dynamic Programming   | -                    | - (23:59 CET)         | TBD                 |
| 05              | Steering (Dubins)     | -                    | - (23:59 CET)         | TBD                 |
| 06              | Collision Checking    | -                    | - (23:59 CET)         | TBD                 |

* Most of these exercises are already accessible before the official evaluation opens.
  You are free to engage with them but be aware that changes may occur until the official opening.
  After the evaluation opens the exercise is "frozen" and you can solve it and submit it.

#### **Finals** exercises, graded.

These will be solved in small groups of 3-4 people but the modality is the same as the weekly exercises. 
They will be a bit more involved content-wise.
We will provide instructions on the group forming modality later on.

| Exercise **ID** | **Topic** | Evaluation **opens** | Evaluation **closes** |
|-----------------|-----------|----------------------|-----------------------|
| 09              | ???       | -                    | -                     |
| 10              | ???       | -                    | -                     |

### **Extra** exercises from previous years, not graded.

You can have a look at the finals from last year and challenge yourself.

| Exercise **ID** | **Topic**     | Evaluation **opens** | Evaluation **closes** |
|-----------------|---------------|----------------------|-----------------------|
| 07              | Optimization  | -                    | -                     |
| 08              | Driving Games | -                    | -                     |

## Plagiarism and dishonest conduct

We take plagiarism and dishonest conduct very seriously.
The exercises are meant to be solved **individually**.
We will check for plagiarism with automated software and human help.
Violating
the [ETH plagiarism etiquette](https://ethz.ch/content/dam/ethz/main/education/rechtliches-abschluesse/leistungskontrollen/plagiarism-citationetiquette.pdf)
will result in consequences according to the ETH regulations:

> "C. The consequences of plagiarism Pursuant to Art. 2 Para. b of the ETH Zurich Disciplinary Code (RSETHZ 361.1)
> plagiarism constitutes a disciplinary violation and will result in disciplinary procedures.
> Detailed information regarding these procedures and their jurisdiction may be found in the ETH Zurich Disciplinary
> Code (RSETHZ 361.1 / www.rechtssammlung.ethz.ch)."

## Getting started

We will use:

- [Git](https://git-scm.com/) as version control system;
- [Python](https://www.python.org/) as programming language;
- [Docker](https://www.docker.com/) as environment containerization (but you won't see it);

If they all sound completely new to you **do not panic**.
We will require a very basic use of most of them, but it is a good time to start learning these tools since they are all
widely adopted in modern robotics.

If you get stuck, try to pair with experienced colleagues for help.
When this still does not solve the problem, try to reach out to the instructors on XXX or at the office hour.

If you are using a Linux-based OS the process should be straight forward.
Windows and Mac can give some more hiccups in the setup, but they are supported as well.

### Git

Git is a version control software.
Please find more explanation under the "Resources" paragraph if you have never heard of it before.
You will need Git on your computer and a GitHub account.

#### Installing Git

Simply follow the steps for your OS at [this link](https://git-scm.com/downloads)

### Creating a GitHub account

If you do not have already a GitHub account create a new one [here](https://github.com/join)

#### _Resources_

- [Git introduction](https://docs.duckietown.com/ente/devmanual-software/basics/development/git.html)
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

* (Mac, Linux) [installation instructions](https://docs.docker.com/engine/install/)
* (Windows) the procedure is slightly more complicated, we suggest to start from [here](https://docs.docker.com/desktop/install/windows-install/). 
Make sure to install WSL 2 and Ubuntu 22.04 LTS as reccomended distro.

#### _Resources_

- [Docker intro](https://docs.duckietown.com/ente/devmanual-software/basics/development/docker.html)

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
