
# Exploring Artificial Intelligence

A foray into artificial intelligence, with the help of math, history and Python  
by _Jonty Sinai_

## Section 1: Foundations of Machine Learning
## Part 0: Introduction

I have just finished Andrew Ng's Machine Learning course on Coursera. Having done that course I can confidently say that machine learning isn't that scary afterall. My mathematical fluency is more than good enough to delve into the theory of machine learning. However, the devil is in the details and experience is the greatest currency when it comes to machine learning. It's easy to make mistakes and every mistake is a learning experience. Mastering machine learning (and later artificial general intelligence) is going to require years of dedication: both the theory (mathematics) and the practice (programming). 

This series of Jupyter notebooks - with corresponding blogposts and Github repo - is the next stage in my journey to understanding artificial intelligence. Although much of the work I do here will be my own interpretation of the history and development of AI, this journey would not have been possible without the knowledge and resources that so many others before me have dedicated their time to. In particular, the content of this series is inspired by Andrey Kurenkov's brilliant [series](http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/) on the history of neural networks and deep learning, as well as Sebastian Raschka's highly informative _Python Machine Learning_ [book](https://sebastianraschka.com/books.html).

This may be the beginning, but the journey ahead is enough to be excited about. As Andrew Ng said in a Forbe's [interview](https://www.forbes.com/sites/peterhigh/2017/06/05/ai-influencer-andrew-ng-plans-the-next-stage-in-his-extraordinary-career/#577a807a3a2c):

> The secret to learning is to not do it only for a weekend, but week after week for a year, or week after week for a decade. The time scale is measured in months or years, not in weeks.

I take this extraordinary man's advice to heart: learning isn't measured in _weeks_ but in _years_. The key is to be _deliberate_ and _focused_ and to keep doing more each day. The secret is _believe_ in more, and to strive to _reach higher_ each day. 

***

Jonathan Sinai  
Johannesburg  
July 2017  

## Setting up Python 3 and the SciPy Stack

To get setup on Mac, run the following sequence of commands to install Python 3 and the required packages. Procedure taken from DigitalOcean's [guide](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-macos) for installing Python 3 on macOS. 

First make sure that the _XCode CommandlineTools_ have been setup:

```
xcode-select --install
```

Next install and setup up _Homebrew_:

```
/usr/bin/ruby -e "(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Now setup a _PATH environment variable_ for Homebrew:

```
nano ~/.bash_profile
```

This will open the _Nano_ commandline text editor. In the text editor window, write the following:

```
export PATH=/usr/local/bin:PATH
```
To save the changes, press `ctrl`+`o`, `return` and then `ctrl`+`x`. 

Now _install Python 3_ using Homebrew:

```
brew install python3
```

Next we will need to install the _SciPy_ package stack:

```
pip3 install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```
Finally install _Scikit-learn_:

```
pip3 install --user scikit-learn
```
