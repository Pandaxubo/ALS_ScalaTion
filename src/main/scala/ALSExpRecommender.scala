
import scalation.random.RandomMatD
import scalation.random.VariateMat
import scalation.linalgebra.MatrixD
import scalation.linalgebra.Eigenvalue
import scalation.linalgebra.Eigenvector
import scalation.linalgebra._
import scalation.linalgebra.Fac_LU
import scala.util.Random
import scalation.linalgebra.VectorD
import scalation.analytics.recommender._
import scalation.util._
import scala.math.{abs, round, sqrt}

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/**  The ALSExpRecommender class is used to perform predictions based on
  *  Model based Collaborative Filtering techniques (Explicit ALS)
  *  @param input  original matrix
  *  @param m      number of rows
  *  @param n      number of columns
  */
class ALSExplicitRecommender (input: MatrixI, m: Int, n: Int) extends Recommender{
    private var predicted  = new MatrixD(m, n)                                // Matrix for storing ALS Explicit predicted values
    private val ratings = makeRatings(input, m, n)                              // original ratings matrix
    private var training = new MatrixD(ratings.dim1, ratings.dim2)              // training dataset
    private var copy_training = new MatrixD(ratings.dim1, ratings.dim2)         // copy of training dataset
    private val als = new ALSExplicit(training)
    


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a rating based on Singular value Decompostion of matrix
      *  @param i  user
      *  @param j  item
      */
    def rate (i: Int, j: Int) : Double = predicted(i, j)


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Generates the training matrix for the dataset for Test 1
      *  @param train  training data matrix
      */
    def genTrain2 (train: MatrixI)
    {
        for (i <- train.range1) training(train(i, 0), train(i, 1)) = train(i, 2)
        copy_training = training.copy()
    } 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Run the ALSExplicit algorithm and get the predict matrix.
      */
    def ALSExp{
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
        }// for
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
        }// for
        copy_training = training.copy()
    } //zRatings

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSExpRecommenderTest` object is used to test the `Recommender` Trait using the MovieLens dataset.
  *  > runMain ALSExpRecommenderTest
  */
object ALSExpRecommenderTest extends App{
    
    val BASE_DIR = System.getProperty("user.dir")
    //val data_file =  BASE_DIR + "/data/sorted_data.txt"
    val data_file =  BASE_DIR + "/data/rating.txt"
    //val data_file =  BASE_DIR + "/data/steam_games/target.csv"
    MatrixI.setSp('\t')
    var input   =  MatrixI(data_file)
    input.setCol(0, input.col(0)-1)
    input.setCol(1, input.col(1)-1)
    //val (m, n)   = (943, 1682)
    val (m, n)   = (6040, 3952)
    //val (m, n)   = (11345, 3582)
    println("Finished reading m and n.")
    val rec    = new ALSExplicitRecommender(input, m, n)
    val rating = rec.makeRatings(input, m, n)

    val train_file =  BASE_DIR + "/data/u2Data.train"           // replace u(1-5).base
    val test_file  =  BASE_DIR + "/data/u2Data.test"  
    //val train_file =  BASE_DIR + "/data/steam_games/training.csv"           // replace u(1-5).base
    //val test_file  =  BASE_DIR + "/data/steam_games/testing.csv"           // replace u(1-5).test
    println("Imported training and testing set.")
    var train   =  MatrixI(train_file)
    train.setCol(0, train.col(0)-1)
    train.setCol(1, train.col(1)-1)
    println("Generating training set.")
    var tester   =  MatrixI(test_file)
    tester.setCol(0, tester.col(0)-1)
    tester.setCol(1, tester.col(1)-1)
    println("Generating testing set.")
    rec.genTrain2(train)
    println("Finishing data preprocessing.")

    for(i <- 0 until 1) {
        val t = time{
            println("Training Time")
            val t1 = time {                                         // training time
                rec.ALSExp                                       
            } //time
            println("Prediction Time")                              // testing time
            val t2 = time {
                rec.error_metrics(tester)
            } // time
        } //time
    } // for
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSExpRecommenderTest2` object is used to test the `Recommender` Trait using the MovieLens dataset.
  * Uses K-fold validation and adds HIT value, which means the predict value is the same as original value.
  *  > runMain ALSExpRecommenderTest2
  */
object ALSExpRecommenderTest2 extends App
{
    val BASE_DIR = System.getProperty("user.dir")
    val data_file =  BASE_DIR + "/data/rating.txt"
    //val data_file   =  BASE_DIR + "/data/sorted_data.txt"
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
    //val (m, n)   = (943, 1682)
    val (m, n)   = (6040, 3952)
    val rec = new  ALSExplicitRecommender(input, m, n)
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
                    rec.ALSExp
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
            } // for

            println("MAE            = "+diff.mean)
            println("MAE rounded    = "+rdiff.mean)
            println("RMSE           = "+diff2.mean)
            println("RMSE rounded   = "+rdiff2.mean)
            println("HIT            = "+hit.mean)
        } // time
    } // for
} 


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSImpRecommenderTest3` object is used to test the `Recommender` Trait using the MovieLens dataset.
  * Uses cross validation.
  *  > runMain ALSImpRecommenderTest3
  */
object ALSExpRecommenderTest3 extends App
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
    val rec    = new ALSExplicitRecommender(input, m, n)
    val t_idx    = VectorD.range(INT_START, INTERVALS)

    for (i <- INT_START until INTERVALS) {
        rec.zRatings((i-1) * INT_SIZE, input)                           // get Zeroes Rating matrix
        println(i)
        rec.ALSExp                                              
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