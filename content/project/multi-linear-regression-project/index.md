---
title: Multivariate Linear Regression
summary: Statistical model that uses N parameters to predict the outcome of the response variable. Implemented with the gradient descent aglorithm.
tags:
- Machine Learning
- Supervised ML
date: "2020-05-27T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Screenshot of the Output
  focal_point: Smart

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/georgecushen
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: example
---


> "Some famous quote about something" - someone important

```python
import numpy as np
```
Have you ever had this feeling that bus (or tram, or train, etc.) leaves exactly 1 minute before you get to the stop AND the next one is forever away? Or you come to a crossroad which is usually very quiet, but exactly as you are about to cross it, there are like 100 cars going in every direction?

Apparently, both of those phenomenon’s can be explained with what is called “Hitchiker’s paradox”. To save you a bit of clicking, the upshot is that:

… if you take an interval of observation that is big enough, you will observe as many arrivals below 10 minutes and as many arrivals above 10 minutes. And you will see that the average waiting time is 10 minutes.

I’ve learned about long time ago and as soon as I did I started noticing it everywhere. Not that those things didn’t happen with me before, so it is probably some variant of Baader-Meinhof phenomenon.

Anyway, the point of this post is to check it with “real” data that I’ve collected myself (hence “Quantified-self” in the title). The source of the data is my notes every time I came to the tram stop that I take every day to work. Since it happens at around the same time every day, I thought that it should be quite consistent and all variance is explained by the paradox above. Ultimately, I want to find out whether, in general, arrival of tram to my stop follows the Poisson distribution with \lambdaλ equal to 4 minutes (official interval at the stop).

That all being said, the data is as follows:

library(ggplot2)
stops <- c(4, 2, 6, 4, 0, 5, 1, 3, 2, 3, 1, 2, 3, 4, 3, 4, 7, 5,
           3, 3, 0, 2, 2, 2, 2, 0, 3, 1, 0, 4, 5, 1, 2, 4, 3, 2,
           8, 2, 2, 5, 5)

ggplot(tibble::enframe(stops), aes(x = value)) +
  geom_bar() +
  labs(title = "How long I waited each time",
       x = "Minutes",
       y = "Frequency")


And \lambdaλ is:

parms <- MASS::fitdistr(stops, "poisson")
parms
##     lambda  
##   2.9268293
##  (0.2671817)
So, in fact, whatever feeling I had of trams being too slow to arrive, it looks like they come (on average) even faster than they should. What would I have done without math?

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis posuere tellus ac convallis placerat. Proin tincidunt magna sed ex sollicitudin condimentum. Sed ac faucibus dolor, scelerisque sollicitudin nisi. Cras purus urna, suscipit quis sapien eu, pulvinar tempor diam. Quisque risus orci, mollis id ante sit amet, gravida egestas nisl. Sed ac tempus magna. Proin in dui enim. Donec condimentum, sem id dapibus fringilla, tellus enim condimentum arcu, nec volutpat est felis vel metus. Vestibulum sit amet erat at nulla eleifend gravida.

Nullam vel molestie justo. Curabitur vitae efficitur leo. In hac habitasse platea dictumst. Sed pulvinar mauris dui, eget varius purus congue ac. Nulla euismod, lorem vel elementum dapibus, nunc justo porta mi, sed tempus est est vel tellus. Nam et enim eleifend, laoreet sem sit amet, elementum sem. Morbi ut leo congue, maximus velit ut, finibus arcu. In et libero cursus, rutrum risus non, molestie leo. Nullam congue quam et volutpat malesuada. Sed risus tortor, pulvinar et dictum nec, sodales non mi. Phasellus lacinia commodo laoreet. Nam mollis, erat in feugiat consectetur, purus eros egestas tellus, in auctor urna odio at nibh. Mauris imperdiet nisi ac magna convallis, at rhoncus ligula cursus.

Cras aliquam rhoncus ipsum, in hendrerit nunc mattis vitae. Duis vitae efficitur metus, ac tempus leo. Cras nec fringilla lacus. Quisque sit amet risus at ipsum pharetra commodo. Sed aliquam mauris at consequat eleifend. Praesent porta, augue sed viverra bibendum, neque ante euismod ante, in vehicula justo lorem ac eros. Suspendisse augue libero, venenatis eget tincidunt ut, malesuada at lorem. Donec vitae bibendum arcu. Aenean maximus nulla non pretium iaculis. Quisque imperdiet, nulla in pulvinar aliquet, velit quam ultrices quam, sit amet fringilla leo sem vel nunc. Mauris in lacinia lacus.

Suspendisse a tincidunt lacus. Curabitur at urna sagittis, dictum ante sit amet, euismod magna. Sed rutrum massa id tortor commodo, vitae elementum turpis tempus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean purus turpis, venenatis a ullamcorper nec, tincidunt et massa. Integer posuere quam rutrum arcu vehicula imperdiet. Mauris ullamcorper quam vitae purus congue, quis euismod magna eleifend. Vestibulum semper vel augue eget tincidunt. Fusce eget justo sodales, dapibus odio eu, ultrices lorem. Duis condimentum lorem id eros commodo, in facilisis mauris scelerisque. Morbi sed auctor leo. Nullam volutpat a lacus quis pharetra. Nulla congue rutrum magna a ornare.

Aliquam in turpis accumsan, malesuada nibh ut, hendrerit justo. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Quisque sed erat nec justo posuere suscipit. Donec ut efficitur arcu, in malesuada neque. Nunc dignissim nisl massa, id vulputate nunc pretium nec. Quisque eget urna in risus suscipit ultricies. Pellentesque odio odio, tincidunt in eleifend sed, posuere a diam. Nam gravida nisl convallis semper elementum. Morbi vitae felis faucibus, vulputate orci placerat, aliquet nisi. Aliquam erat volutpat. Maecenas sagittis pulvinar purus, sed porta quam laoreet at.
