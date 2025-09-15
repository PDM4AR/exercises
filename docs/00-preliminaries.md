# Preliminaries
{:.no_toc}

<div markdown="block">
  <summary>
    Table of contents
  </summary>
1. TOC
{:toc}
</div>

---

## Rules of the game

The programming exercises are a great way to understand and apply what you learn in class.

There are four types of exercises:

- **Warm-up** exercise(s), not graded;
- **Weekly** individual exercises, graded (will account for 30% of the final grade);
- **Finals** group exercises, graded (will account for 70% (30 + 40) of the final grade);
- **Extra** exercises from previous years, not graded.

#### **Warm-up** exercises, not graded

Just ensuring everything is working as expected.

#### **Weekly** individual exercises, graded

The weekly exercises must be solved *individually*.
They will involve relatively short programming tasks related to the topics covered in class.
You will submit your solutions via your private GitHub repository. Detailed submission instructions are provided later.

**Important information**:

* Each exercise has a limited submission window for 2-3 weeks.
* Only the last **VALID** submission (i.e., processed by the server regardless of the outcome) within the exercise time window counts toward your exercise grade. A missing submission counts as 0.
* Your score will be determined by the performance metrics specified in the exercise description. While multiple metrics are considered, **CORRECTNESS** is the most important factor. For students who wish to pursue higher performance(e.g. in solving time), we provide reference values based on the TA solution and/or historical data. The mapping from scores to grades will **NOT** be disclosed in advance.
* Only your best **_N-1_** exercise results contribute to 30% of your final grade, where _N_ is the total number of graded weekly exercises (N=5).
* For each exercise, you have a limited number of server submissions. You can test locally as much as you want. Consider this a learning experience for robotics; field tests are expensive.
* Here you can see a temporary schedule (will be completed on the fly):

| Exercise **ID** | **Topic**             | Evaluation **opens** | Evaluation **closes**        | **Deadline status** | Available Submissions |
|-----------------|-----------------------|----------------------|------------------------------|---------------------|-----------------------|
| 02              | Graph Search          | 24th of September    | 14th of October (23:59 CET)  | tentative           | 10                    | 
| 03              | Informed Graph Search | 24th of September    | 14th of October (23:59 CET)  | tentative           | 10                    |
| 04              | Dynamic Programming   | 1st of October       | 21st of October (23:59 CET)  | tentative           | 10                    |
| 05              | Steering (Dubins)     | 15th of October      | 28th of October (23:59 CET)  | tentative           | 10                    |
| 06              | Collision Checking    | 29th of October      | 11th of November (23:59 CET) | tentative           | 10                    |

* Most of these exercises are already accessible before the official evaluation opens.
  You are free to engage with them earlier, but be aware that changes may occur up until the official opening.
  After the evaluation opens, the exercise is "frozen", and you can solve and submit it.

#### **Finals** exercises, graded

These are solved in small groups of **up to 3 people**, with the same modality as the weekly exercises.
They will be a bit more involved content-wise.
Instructions on group formation are provided via Piazza.

| Exercise **ID** | **Topic**       | Evaluation **opens** | Evaluation **closes**        | **Deadline status** | Available Submissions |
|-----------------|-----------------|----------------------|------------------------------|---------------------|-----------------------|
| 13              | To be announced | 12th of November     | 2nd of December (23:59 CET)  | tentative           | 10                    |
| 14              | To be announced | 3rd of December      | 23rd of December (23:59 CET) | tentative           | 10                    |

#### **Extra** exercises from previous years, not graded

You can have a look at the finals from last year and challenge yourself.

| Exercise **ID** | **Topic**            | Evaluation **opens** | Evaluation **closes** |
|-----------------|----------------------|----------------------|-----------------------|
| 07              | Optimization         | -                    | -                     |
| 08              | Driving Games        | -                    | -                     |
| 09              | PDM4ARocket Explorer | -                    | -                     |
| 10              | Robot Runners        | -                    | -                     |
| 11              | Spaceship            | -                    | -                     |
| 12              | Highway Driving      | -                    | -                     | 

## Plagiarism and dishonest conduct

We take plagiarism and dishonest conduct very seriously.
The exercises are meant to be solved **individually** and **without excessive reliance on AI tools**.
We will check for plagiarism with automated software and human help.
Violating the [ETH plagiarism etiquette](https://ethz.ch/content/dam/ethz/main/education/rechtliches-abschluesse/leistungskontrollen/plagiarism-citationetiquette.pdf) will result in disciplinary actions as per ETH regulations.
> "C. The consequences of plagiarism Pursuant to Art. 2 Para. b of the ETH Zurich Disciplinary Code (RSETHZ 361.1)
> plagiarism constitutes a disciplinary violation and will result in disciplinary procedures.
> Detailed information regarding these procedures and their jurisdiction may be found in the ETH Zurich Disciplinary
> Code (RSETHZ 361.1 / www.rechtssammlung.ethz.ch)."

## Getting started

We will use:

- [Python](https://www.python.org/) as programming language;
- [Docker](https://www.docker.com/) as environment containerization (but you won't see it);
- [Git](https://git-scm.com/) as version control system;
- [GitHub](https://github.com/) as code hosting and assignment distribution and submission platform;
- [VS Code](https://code.visualstudio.com/) as code editor.

If these tools are new to you, do not panic. We require only very basic use of most of them. It's a good time to start learning these tools, as they are widely adopted in modern robotics.

If you get stuck, pair up with experienced colleagues. If that doesn't solve the problem, reach out to the instructors on Piazza or during office hours.

If you are using a Linux-based OS, the process should be straightforward.
Windows and macOS may have more hiccups, but they are supported as well.

### Python

Python will be the programming language adopted in this course. You don't need to install Python on your local machine because you'll work in a Docker container with Python preinstalled (more information below).

#### _Resources_
If you want to learn Python basics, here are some resources to get you started:

- [Python Tutorial](https://www.tutorialspoint.com/python/index.htm), in particular make sure to go through the basic
  data structures (tuples, lists, dictionaries, sets,...), loops (while, for,...), conditional statements (if-else),
  functions, classes and objects.
- [Official documentation](https://docs.python.org/3/)

### Docker

We will run the exercises in a containerized environment, a technique widely used in modern software development. Read the [Resources](#docker-resources) section for more information on Docker.

Now let's install it on your computer:

If you are using __Linux__, you can choose to either install `Docker Engine` or `Docker Desktop`, but __NOT__ both. `Docker Engine` is the core container runtime and is bundled with the Docker CLI `docker`, while `Docker Desktop` is `Docker Engine + GUI + extra tools`.

👉 For **Linux** users of this course, `Docker Engine` is recommended for its simplicity and lower resource usage.

- [Install Docker Engine on Linux](https://docs.docker.com/engine/install/), or
- [Install Docker Desktop on Linux](https://docs.docker.com/desktop/install/linux-install/)

If you are using __Mac__, follow the instructions below to install Docker Desktop:
- [Install Docker Desktop on Mac](https://docs.docker.com/desktop/install/mac-install/)

If you are using __Windows__, follow this YouTube video to install Docker Desktop:
- [Install Docker Desktop on Windows](https://youtu.be/ZyBBv1JmnWQ?si=l6Xfqq7lLQpGrohe)

#### _Resources_
{: #docker-resources}

If you want to learn more about Docker, check out the following resources:
- [Docker intro](https://docs.duckietown.com/ente/devmanual-software/basics/development/docker.html)

#### **Checklist**

- [ ] I have installed Docker on my computer
- [ ] I can run without errors the Docker hello-world (`docker run hello-world`)

### Git & GitHub

Git is a version control software. GitHub is a cloud-based platform that hosts Git repositories. Please see the [Resources](#git-resources) section if you are new to them.
You will need Git on your computer and a GitHub account.

#### Creating a GitHub account

If you do not already have a GitHub account create a new one [here](https://github.com/join).

#### Setting up Git

For __Linux__ or __Mac__ users: 
- Download and install Git for your OS from [this site](https://git-scm.com/downloads). 
- Set up Git and authenticate with GitHub following the instructions [here](https://docs.github.com/en/get-started/git-basics/set-up-git). You are free to pick the connecting and credential caching options that you are most comfortable with.

For __Windows__ users,
- Download and install Git for Windows from [here](https://git-scm.com/downloads/win), which comes with the Git Credential Manager (GCM).
- The simplist way to connect to GitHub on Windows is over HTTPS and use GCM to store your Git credentials. See [here](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git?platform=windows#git-credential-manager) for detailed instructions. 

Notes for __Windows__ users:
- VS Code with Dev Container extension(see below) automatically forwards your cached credential from Windows to the container. This means you can perform all Git operations both on Windows and inside the container without re-authenticating.
- If you clone the repository on Windows (as instructed above), remember that our Linux-based container will access the files through the Windows filesystem. This causes significant overhead for file-intensive operations (e.g., I/O, dependency installation). **For this course, the impact is negligible since no bulky I/O operations are expected.**
- Cloning the repository inside WSL instead of Windows avoids the overhead for file-intensive operations. It requires configuring GCM on WSL separately. If you are interested in following this approach (not necessary for this course). instructions are available [here](https://github.com/git-ecosystem/git-credential-manager/blob/release/docs/wsl.md#configuring-wsl-with-git-for-windows-recommended).

#### _Resources_
{: #git-resources}

- [Git Guide](https://github.com/git-guides)
- [GitHub tutorial](https://guides.github.com/activities/hello-world/)

#### **Checklist**

- [ ] I can run `git version` on my terminal without errors.
- [ ] I have a GitHub account.
- [ ] I know how to use basic git commands `git clone`, `git add`, `git commit` and `git push`.

### VS Code

Using an [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) is not necessary.
But it provides a good set of tools that speeds up development (code navigation, debugging,...).
Moreover, we will provide environment configurations that will make your life easier.

We recommend using VS Code. You can find installation instructions [here](https://code.visualstudio.com/docs/setup/setup-overview). After installing VS Code, also install the Dev Containers extension from [here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

There are many other good IDEs for Python (PyCharm, Atom, Spyder,...) that should work just fine if you know how to replicate the development environment in `.devcontainer/` exactly. We won't support them officially.

## Support

Use the forums on Piazza for general questions: this way, you can help other students who experience the same issues.
