# Crime in Communities: Predicting Violent Crime by County (Econ 187)
### by Jacob Titcomb



* This repository is the second project for Econ 187 (Machine Learning) at UCLA with Professor Randall Rojas, for Spring 2024.

* All work is my own.

* R was the primary language used for this project.

* The data comes from the FBI crime database and US census, via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized).

* My raw data cleaning and model fitting is in the source file, `econ187_proj2_source.R`. The Rmarkdown file `econ187_proj2.Rmd` has the formatting and structuring of the report `econ187_proj2.pdf`. I recommend focusing on either the source file or the final PDF, as those give the most information about the methodology/findings.

* The primary purpose of this project is to provide policy-makers with a more nuanced view of the factors---particularly economic---which characterize violent crime. This project is meant as a *descriptive* tool, rather than a *prescriptive* guide to violent crime reduction.

## Models

This machine learning project focused on modeling violent crime rates at the county level, with statistical/economic inference as the primary goal. For each model I extracted variable importance in order to determine which features have high predictive power with regard to crime.

I fit the following models:

1. Ordinary least squares (OLS)
2. Least Absolute Shrinkage and Selection Operator (LASSO) regression
3. Ridge regression
4. Elastic net regression
5. Principal component regression (PCR)
6. Piece-wise polynomial
7. Multivariate Adaptive Regression Splines (MARS)
8. Generalized Additive Model (GAM)
9. Gaussian Process Regression (GPR)
10. Bayesian ridge regression

## Ethical Considerations

**Please read the project's introduction and conclusion for more thorough, yet still brief, overviews of the ethical issues which arise in this project.**

It goes without saying that finding associations between crime and census data can quickly approach sensitive territory when care is not taken in analysis. The use of machine learning and AI in policing can be particularly harmful to marginalized communities, especially when considering bias within model training.

Producing a model of crime that can deal with explicit and implicit bias is well beyond the scope of this project. Future study in AI bias is necessary across the board, especially when those models can impact the lives of people as in the case with crime data.

As I state in my project, all policies come with a price, both social and economic. Before decisions are made and lives are changed, it is vital to *understand* the issues at hand. This is where I hope my project can contribute.