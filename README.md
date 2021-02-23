# ALS_ScalaTion
Implementation of Alternating Least Squares Algorithm(Implicit Data) by ScalaTion

About ScalaTion:
ScalaTion is a Scala-based system for Simulation, Optimization and Analytics. For here we use the 1.6 version with DEBUG = false. [[project]](http://cobweb.cs.uga.edu/~jam/scalation.html)

About ALS algorithm:
"Collaborative Filtering for Implicit Feedback Data". [[paper]](http://yifanhu.net/PUB/cf.pdf)  

The testing dataset is MovieLens 100k. We use MAE and RMSE to judge the quality.

For recommender algorithm:
To run the code:(In bash)
sbt 
compile 
runMain CFRecommenderTest

Also you can just run ALSTest, and that is a pure object for Matrix Factorization:
To run the code:(In bash)
sbt 
compile 
runMain ALSTest
