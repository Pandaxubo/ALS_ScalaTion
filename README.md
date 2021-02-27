# ALS_ScalaTion
Implementation of Alternating Least Squares Algorithm(Implicit Data) by ScalaTion. 

Environment: Scala(2.12.13) + sbt(1.4.7) + ScalaTion(1.6, all packages).

## About ScalaTion:
ScalaTion is a Scala-based system for Simulation, Optimization and Analytics. For here we use the 1.6 version with DEBUG = false. [[project]](http://cobweb.cs.uga.edu/~jam/scalation.html)

## About ALS algorithm:
"Collaborative Filtering for Implicit Feedback Data". [[paper]](http://yifanhu.net/PUB/cf.pdf)  

## About dataset:
The testing dataset is MovieLens 100k(sorted_data.txt, u2.base, u2.test). [[data]](https://grouplens.org/datasets/movielens/100k/) We use MAE and RMSE to judge the quality.

## Run the code:
### For recommender algorithm:
To run the code:(In bash)
```bash
sbt 
compile 
runMain CFRecommenderTest
```

### Also you can just run ALSTest, and that is a pure object for Matrix Factorization:
To run the code:(In bash)
```bash
sbt 
compile 
runMain ALSTest
```
