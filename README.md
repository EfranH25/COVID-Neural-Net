<br />
<p align="center">
  <h3 align="center">COVID-19 Forecasting Project</h3>

  <p align="center">
    This project attempts to forecast COVID-19 sate outcomes through a machine learning SIR model. Unfortuntely due to insufficent data and scheduling conflicts the project never came to full fruition. Regardless, it was an excellent learning experince to get aquainted with data collection, Neural Nets and Forecasting. Because of this, I thought it would be useful to put my experience and the process on GitHub.
    <br />
    <a href="https://github.com/EfranH25/COVID-Neural-Net"><strong>Explore the Repo »</strong></a>
    <a href="https://github.com/EfranH25/COVID-Neural-Net/blob/master/COVID%2019%20Forecasting%20Report.pdf">View Full Paper</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Abstract](#Abstract)
* [Introduction](#Introduction)
* [Methods](#Methods)
* [Results](#Results)
  * [Multiple_Linear_Regression](#Multiple_Linear_Regression)
  * [Naive_Beta_Estimation](#Naive_Beta_Estimation)
  * [Basic_SIR_Model](#Basic_SIR_Model)
  * [Neural_Nets](#Neural_Nets)
  * [Our_SIR_Model](#Our_SIR_Model)
* [Issues_and_Summary](#Issues_and_Summary)
* [Contact](#contact)

<!-- Abstract -->
## Abstract
The most common model that is used for modeling the progression of infectious diseases is the SIR model. This model tracks the number of susceptible, infected, and recovered individuals in a closed population. Because this SIR model is for a closed population, it does not take into account immigration and emigration as well as geographic components. To address the weaknesses of the basic SIR model, we decided to build our own SIR model, and we hypothesized that our model would provide more accurate forecasting of the morbidity and mortality due to COVID-19 than the basic SIR model would provide. In approximating the rate of a susceptible-infected contact resulting in a new infection and the rate that an infected recovers and moves into the resistant phase, the predicted values that the neural nets produced had very high residuals, so it was difficult to reintroduce the predictions back into our SIR model. 


**Keywords:**
  COVID-19, SIR model, Multiple Linear Regression, Neural Nets

<!-- Introduction -->
## Introduction
The coronavirus pandemic has been the most central topic in the world over the past few months. It has impacted every country in the world, and the cases and deaths due to this virus continue to rise. The United States of America has become an epicenter of this pandemic, and as of April 10, COVID-19 has infected 460,287 people and has killed 16,535 deaths in America [2]. With such a devastating tragedy at the center of our lives right now, our group took a tremendous interest in devoting our project to building a model that would help forecast COVID-19 in the US.

Different countries have responded to the outbreak of COVID-19 in different ways. South Korea was one of the first countries that was able to effectively contain the spread of COVID-19 by installing public phone booths outside of hospitals. At these phone booths, people getting tested for the virus are on one side of the glass, and a hospital worker is on the other side. The hospital worker can swab the patient quickly, disinfect the booth, and get a result within seven minutes. These phone booths allow almost ten times as many samples to be tested. In Senegal, as of March, hospital workers were using test kits to diagnose patients within 4 hours, which is faster than the kits in the U.S. were able to make diagnoses. In comparison to South Korea, countries like Italy and the U.S. responded too moderately and set social distancing measures once the virus had already spread too much [1]. Social distancing measures have been especially important to implement because COVID-19 is primarily spread directly through contact with infected people (in the form of respiratory droplets) or indirectly with surfaces that have been touched by infected people [3]. 

<!-- Methods -->
## Methods
Please view the full paper on the GitHub page to see an more in-depth explanation of the methodologies and process involved (Unfortunately converted the equations and results into GitHub readme proved a bit cumbersome). 

Link: [Full Paper](https://github.com/EfranH25/COVID-Neural-Net/blob/master/COVID%2019%20Forecasting%20Report.pdf)

Topics covered for methods include:
-	Building the Database
-	Multiple Linear Regression
-	Naive Beta Estimation
-	Basic SIR Model
-	Neural Nets
-	Neural Nets
-	Simulation
-	Beta Estimation

<!-- Results -->
## Results
### Multiple Linear Regression
After checking the no multicollinearity assumption and removing the highly correlated variables from the multiple linear regression model, we found that the only statistically significant coefficients were those for the population density and number of airports predictor variables. As expected, the coefficients for population density and number of airports were positive (21.4757 and 12.5072) for the model predicting COVID-19 cases and were positive (0.7089 and 0.3736) for the model predicting COVID-19 deaths. As population density increases, there are likely more cumulative interactions between humans, and as a result, the transmission of the virus from the infected to the susceptible is easier, increasing the number of COVID-19 cases and potential deaths. Similarly, as the number of airports increases, the amount of interstate travel in the U.S. increases, so it is more likely that someone from New York (a state with the most COVID-19 cases and deaths) comes into contact with someone from Alaska (a state with only a few hundred COVID-19 cases). Thus, the transmission of the virus becomes easier, increasing the number of COVID-19 cases and potential deaths. After checking the linearity and normality assumptions, the only variable left was the number of airports. The number of airports was the only predictor variable that satisfied the three assumptions.

### Naive Beta Estimation
We took a somewhat Bayesian approach was taken for estimating Beta using our model.

Firstly, there is the problem of estimating the true data (susceptible, infected, etc) given the reported data. Given the true data, the reported would follow a hypergeometric distribution, and so the distribution of the true given the reported can be expected to follow a negative hypergeometric distribution. For ease of computation this was approximated by negative binomial distributions instead, which is accurate in the limit of a large unexposed population.

For a binomial distribution with (EQUATION IN FULL PAPER), the Jeffreys prior can be calculated to be (EQUATION IN FULL PAPER), which we used as a very rough estimate of the Jeffreys prior for (EQUATION IN FULL PAPER), and sampled from. This was done by applying its quantile function (EQUATION IN FULL PAPER) to equally-spaced values on [0,1].

What was then needed was to calculate the likelihood of the data given the different values of Beta. To do this, for each time except the last, the next was simulated, estimating the appropriate conditional distributions. In order to manage the dimensionality of the problem, the different locations were considered as to be were independent. Likewise with the infected and recovered groups; the susceptible group was disregarded since it is by necessity linearly dependent on the other two.

After normalization this produced a distribution over Beta, from which the mean and standard deviation could be extracted.

### Basic SIR_Model
After omitting the dynamics of birth and death from our basic SIR model, we found that the basic SIR model was able to predict the parameters for the United States more accurately. Our model predicted a beta (the infection rate of the disease) value of 0.1496 and a gamma (the recovery rate) value of 0.002533. Our gamma value seems to be an underestimate, which is most likely because of the sparsity of recovery rate data. Due to our gamma value being lower than expected, the model produced a reproduction rate ($R_0$) of approximately 59.078, which is approximately ten times higher than it should be.

### Neural Nets
Our initial plan was to perform predictions for several time frames to get an idea of the rate of infection and rate of deaths per state to then feed into our SIR model. Unfortunately, after some initial training runs, we encountered a major roadblock: the predicted values in our trained neural nets had extremely high residuals. We encountered residuals as high as 18,000 for the mean absolute error for estimated positive cases, and these high residuals prevented us from proceeding further onto our final goal of estimated infection and death rates. We tried different approaches to optimizing our neural nets, most of them by trial and error and cycling from different iterations ranging from models with very few nodes with only one hidden neural net cluster to hundreds of nodes with multiple hidden clusters. Eventually, after these iterations were performed, we found a neural net with one hidden cluster and a moderate number of nodes, 64 per cluster, to be acceptable because it allowed us to reduce our residuals to around 5,900 with a testing training split of 80/20. We found this residual to be acceptable because we realized that many states in the U.S. had been under-testing their populations, so the true number of actual cases was likely much higher than reported. Unfortunately, by the time we realized this, our presentation deadline was approaching, so instead of refining our model even further to eventually predict the infection rate and deaths, we instead proceeded to focus our attention in completing our SIR model. 

### Our SIR Model
Please view the full paper on the GitHub page to view our model. (Again, unfortunately GitHub made it difficult to post our results on the ReadMe) 

Link: [Full Paper](https://github.com/EfranH25/COVID-Neural-Net/blob/master/COVID%2019%20Forecasting%20Report.pdf)

<!-- Issues and Summary -->
## Issues and Summary
We will outline a few potential issues that need to be resolved so that our model can be more accurate. The first issue we will discuss is that our model does not account for many of the factors that contribute to the spread of COVID-19, such as the effects of social distancing. Secondly, we found that multiple linear regression does not work well for COVID-19 forecasting. A potential solution would be to use logistic regression with a binary variable for survival/death from COVID-19. Lastly, in order to have an accurate model, we need reliable data. The data we collected were from various reliable sources. A background check on the data and how it was collected could help improve our model. 
While our model does have some issues, it was still helpful in making some significant findings. Through our regression model, we were able to find that the number of airports and population density supposedly play an important role in predicting the number of deaths and total cases due to COVID-19. We also found that neural nets have a lot of potential to be useful in forecasting COVID-19. Through our own SIR model, we found that compared with a basic SIR model, our model shows jumps in the infection curve as a new population is infected. 
The main limitation for our SIR model was the inaccuracy of our neural nets. For future models, we will have to reconfigure our neural nets in order to fully explore the SIR model. Moving forward, we will estimate the infection and recovery rates for the model with our current neural nets and compare to real world outcomes. We will also create a secondary model that uses actual data provided by state databases to create another SIR model based on real world values instead of predicted values. 
As the world continues to fight against the COVID-19 pandemic, reliable statistical models are needed to accurately predict the spread of infection. Accurate predictions will help inform world leaders to make the proper decisions for their respective country. With the improvements listed, our hope is that the model we created can be useful in aiding in the fight against COVID-19.

<!-- Final Model -->
## Final Model

<!-- CONTACT -->
## Contact

Efran Himel - efranhimel@gmail.com

Project Link: [https://github.com/EfranH25/COVID-Neural-Net](https://github.com/EfranH25/COVID-Neural-Net)
