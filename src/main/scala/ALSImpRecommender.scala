import scalation.linalgebra.MatrixI
import scalation.linalgebra.MatrixD
import scalation.linalgebra.VectorI
import scalation.linalgebra.VectorD

import scalation.analytics.recommender._
import scalation.util.time
import scala.math.{abs, round, sqrt}

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/**  The ALSImpRecommender class is used to perform predictions based on
  *  Model based Collaborative Filtering techniques (Implicit ALS)
  *  @param input  original matrix
  *  @param m      number of rows
  *  @param n      number of columns
  */
class ALSImpRecommender (input: MatrixI, m: Int, n: Int) extends Recommender{
    private var predicted = new MatrixD(m, n)                                // Matrix for storing Implicit ALS predicted values
    private val ratings = makeRatings(input, m, n)                              // original ratings matrix
    private var training = new MatrixD(ratings.dim1, ratings.dim2)              // training dataset
    private var copy_training = new MatrixD(ratings.dim1, ratings.dim2)         // copy of training dataset
    private val als = new ALSImplicit(training)


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a rating based on Implicit ALS of matrix
      *  @param i  user
      *  @param j  item
      */
    def rate (i: Int, j: Int) : Double = predicted(i, j)


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Generates the training matrix for the dataset for Test 1
      *  @param train training data matrix
      */
    def genTrain2 (train: MatrixI)
    {
        for (i <- train.range1) training(train(i, 0), train(i, 1)) = train(i, 2)
        copy_training = training.copy()
    } 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Run the ALSImplicit algorithm and get the predict matrix.
      */
    def ALSImp{
        predicted = als.train()
    }


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Generates the training matrix for the dataset for Test 2
      *  @param exl  vector of index values which will be excluded in the train
      *  @param input  original data matrix
      */
    def genTrainTest (exl: VectorI, input: MatrixI): MatrixD =
    {
        for (i <- input.range1){
            if(exl.indexOf(i) != -1) training(input (i, 0), input (i, 1)) = input(i, 2)
            else training(input (i, 0), input (i, 1)) = 0.0
        }
        copy_training = training.copy()
        ratings - training
    } 
            
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Returns a matrix with replacing all values with 0 beyond the interval specified
      *  corresponds to Test 3 of the expermiments
      *  @param limit  interval start point
      *  @param input  original data matrix
      */
    def zRatings (limit : Int, input: MatrixI)
    {
        for (i <- input.range1){
            if(i <= limit) training(input (i, 0), input (i, 1)) = input(i, 2)
            else training(input (i, 0), input (i, 1)) = 0.0
        }
        copy_training = training.copy()
    } 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test 1: Print MAE and RMSE metrics based on the final predictions for
     *  the test dataset.
     *  This function needs to rewrite since we need to calculate MPR value.
     *  @param input  the test portion of the original 4-column input matrix
     */
    def error_metric (input: MatrixI)
    {   
       var sum1, sum2, sum3, sum4, sum5, sum6 = 0.0
       var cp = predicted.copy()
       cp = cp - cp.min(cp.range1, cp.range2)
       cp = cp/cp.max(cp.range1, cp.range2)
        for (i <- input.range1) {
            val a = input(i, 2).toDouble
            val p = rate (input(i, 0), input(i, 1))
            if (! p.isNaN) {
                sum1 += abs (a - p)                               // non rounded MAE
                sum2 += abs (a - round (p))                       // rounded MAE
                sum3 += (a - p) * (a - p)                         // non rounded RMSE
                sum4 += (a - round (p)) * (a - round (p))         // rounded RMSE
                sum5 += a*cp(input(i, 0), input(i, 1))
                sum6 += a
            } // if
        } //for
        println ("MAE  Non-Rounded = " + sum1 / input.dim1)
        println ("MAE      Rounded = " + sum2 / input.dim1)
        println ("RMSE Non-Rounded = " + sqrt (sum3 / input.dim1))
        println ("RMSE     Rounded = " + sqrt (sum4 / input.dim1))
        println ("Predict precentage max: "+cp.max(cp.range1, cp.range2)+", min: "+cp.min(cp.range1, cp.range2))
        println ("MPR Non-Rounded = " + sqrt (sum5 / sum6))
    } // error metrics
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSImpRecommenderTest` object is used to test the `Recommender` Trait using the MovieLens dataset.
  *  > runMain ALSImpRecommenderTest
  */
object ALSImpRecommenderTest extends App{
    val BASE_DIR = System.getProperty("user.dir")
    val data_file =  BASE_DIR + "/data/sorted_data.txt"
    MatrixI.setSp('\t')
    var input   =  MatrixI(data_file)
    input.setCol(0, input.col(0)-1)
    input.setCol(1, input.col(1)-1)
    val (m, n)   = (943, 1682)
    val rec    = new ALSImpRecommender(input, m, n)
    val rating = rec.makeRatings(input, m, n)

    val train_file =  BASE_DIR + "/data/u2.base"           // replace u(1-5).base
    val test_file  =  BASE_DIR + "/data/u2.test"           // replace u(1-5).test

    var train   =  MatrixI(train_file)
    train.setCol(0, train.col(0)-1)
    train.setCol(1, train.col(1)-1)

    var tester   =  MatrixI(test_file)
    tester.setCol(0, tester.col(0)-1)
    tester.setCol(1, tester.col(1)-1)

    rec.genTrain2(train)


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Implicit ALS needs binary rating matrix as actual value
      *  @param a  testing matrix
      */
    def ConTestP(a: MatrixI) : MatrixI = {
        val P = a.copy()
            for (i <- a.range1 ) {
                    if(P(i,2) > 0)  
                        P(i,2) = 1
                }
        P
    }

    tester = ConTestP(tester)


    for(i <- 0 until 1) {
        val t = time{
            println("Training Time")
            val t1 = time {                                         // training time
                rec.ALSImp                                       // Implicit ALS
            } //time
            println("Prediction Time")                              // testing time
            val t2 = time {
                rec.error_metric(tester)
            } // time
        } //time
    } // for
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSImpRecommenderTest2` object is used to test the `Recommender` Trait using the MovieLens dataset.
  * Uses K-fold validation and adds HIT value, which means the predict value is the same as original value.
  *  > runMain ALSImpRecommenderTest2
  */
object ALSImpRecommenderTest2 extends App
{
    val BASE_DIR = System.getProperty("user.dir")
    val data_file   =  BASE_DIR + "/data/sorted_data.txt"
    val kfold       = 5                                                              // value for k-fold cross-validation
    val diff        = new VectorD(kfold)                                             // mae values
    val rdiff       = new VectorD(kfold)                                             // rounded mae values
    val diff2       = new VectorD(kfold)                                             // rmse value
    val rdiff2      = new VectorD(kfold)                                             // rounded rmse values
    val hit         = new VectorD(kfold)                                             // number of successful predictions

    MatrixI.setSp('\t')
    var input   =  MatrixI(data_file)
    input.setCol(0, input.col(0) -1)
    input.setCol(1, input.col(1) -1)

    val foldsize    = input.dim1/kfold
    val (m, n)   = (943, 1682)
    val rec = new ALSImpRecommender(input, m, n)
    for(x <- 0 until 1) {
        val t = time {
            val indx = List.range(0, input.dim1)
            val rand_index = scala.util.Random.shuffle(indx)
            val index = new VectorI(input.dim1)
            val fold = VectorD.range(0, kfold)
            for (i <- 0 until input.dim1) index(i) = rand_index(i)              // create a vector of a randomly permuted matrix

            for (i <- 0 until kfold) {
                val excl = new VectorI(foldsize)                                // Vector to track the exclusion ratings
                println(s"--------------$i------------------")
                for (j <- 0 until excl.dim)
                    excl(j) = index(i * foldsize + j)
                val tester = rec.genTrainTest(excl, input)
                println("Training Time")
                val t1 = time {                                                 // training time
                    rec.ALSImp                                          
                }
                println("Prediction time")
                val t2 = time {                                                 // prediction time
                    rec.crossValidate(tester)
                }
                val stats = rec.getStats
                diff(i) = stats(0).ma
                rdiff(i) = stats(1).ma
                diff2(i) = stats(0).rms
                rdiff2(i) = stats(1).rms
                hit(i) = stats(2).mean * 100
                for (j <- 0 until 3) stats(j).reset()
            } 

            println("MAE            = "+diff.mean)
            println("MAE rounded    = "+rdiff.mean)
            println("RMSE           = "+diff2.mean)
            println("RMSE rounded   = "+rdiff2.mean)
            println("HIT            = "+hit.mean)
        } 
    } 
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSImpRecommenderTest3` object is used to test the `Recommender` Trait using the MovieLens dataset.
  * Uses cross validation.
  *  > runMain ALSImpRecommenderTest3
  */
object ALSImpRecommenderTest3 extends App
{
    val BASE_DIR = System.getProperty("user.dir")
    val data_file =  BASE_DIR + "/data/sorted_data.txt"
    val kfold  = 5                                                              // value for k-fold cross-validation
    val INTERVALS = 100                                                         // time intervals for observation of statistics
    val INT_SIZE  = 1000                                                        // no of ratings in each interval
    val INT_START = 75                                                          // starting point of the interval
    val diff  = new VectorD(INTERVALS - INT_START)                              // MAE
    val rdiff = new VectorD(INTERVALS - INT_START)                              // MAE rounded
    val diff2  = new VectorD(INTERVALS - INT_START)                             // RMSE
    val rdiff2 = new VectorD(INTERVALS - INT_START)                             // RMSE rounded
    val hit   = new VectorD(INTERVALS - INT_START)                              // number of successful predictions

    MatrixI.setSp('\t')
    var input   =  MatrixI(data_file)
    input.setCol(0, input.col(0) -1)
    input.setCol(1, input.col(1) -1)

    val (m, n)   = (943, 1682)
    val rec    = new ALSImpRecommender(input, m, n)
    val t_idx    = VectorD.range(INT_START, INTERVALS)

    for (i <- INT_START until INTERVALS) {
        rec.zRatings((i-1) * INT_SIZE, input)                           // get Zeroes Rating matrix
        println(i)
        rec.ALSImp                                                    // Implicit ALS                                   
        rec.test((i-1) * INT_SIZE, i * INT_SIZE, input)
        val stats = rec.getStats
        diff(i-INT_START)   = stats(0).ma
        rdiff(i-INT_START)  = stats(1).ma
        diff2(i-INT_START)  = stats(0).rms
        rdiff2(i-INT_START) = stats(1).rms
        hit(i-INT_START)    = stats(2).mean * 100
        for (j <- 0 until 3) stats(j).reset()
    } // for

    println(diff.mean)
    println(rdiff.mean)
    println(diff2.mean)
    println(rdiff2.mean)
    println(hit.mean)
    println(diff)
    println(rdiff)
    println(hit)

}