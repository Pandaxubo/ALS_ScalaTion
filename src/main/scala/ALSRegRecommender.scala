
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
/**  The ALSRegRecommender class is used to perform predictions based on
  *  Model based Collaborative Filtering techniques (Regularized ALS)
  *  @param input  original matrix
  *  @param m      number of rows
  *  @param n      number of columns
  */
class ALSRegRecommender (input: MatrixI, m: Int, n: Int) extends Recommender{
    private var predicted  = new MatrixD(m, n)                                // Matrix for storing Regularized ALS predicted values
    private val ratings = makeRatings(input, m, n)                              // original ratings matrix
    private var training = new MatrixD(ratings.dim1, ratings.dim2)              // training dataset
    private var copy_training = new MatrixD(ratings.dim1, ratings.dim2)         // copy of training dataset
    private var testing = new MatrixD(ratings.dim1, ratings.dim2) 
    private var copy_testing = new MatrixI(ratings.dim1, ratings.dim2)
    private var copy_train = new MatrixI(ratings.dim1, ratings.dim2)
    private val als = new ALSReg(training)
    


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a rating based on ALS of matrix
      *  @param i  user
      *  @param j  item
      */
    def rate (i: Int, j: Int) : Double = predicted(i, j)


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Generates the training matrix for the dataset for Test 1
      *  @param train  training data matrix
      */
    def genTrain2 (train: MatrixI): MatrixI =
    {
        for (i <- train.range1) training(train(i, 0), train(i, 1)) = train(i, 2)
        copy_train = training.copy().toInt
        copy_train
    }

    def genTest2 (train: MatrixI): MatrixI =
    {
        for (i <- train.range1) testing(train(i, 0), train(i, 1)) = train(i, 2)
        copy_testing = testing.copy().toInt
        copy_testing
    } 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Run the ALSExplicit algorithm and get the predict matrix.
      */
    def ALSReg(I: MatrixD, I2: MatrixD): MatrixD = {
        var predicted = als.train(I, I2)
        predicted
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



    def getNonZero(testing: MatrixI): Double = {
        var T_vector = VectorD(0.0)
        for(i <- testing.range1){
            // var a = new VectorD(testing.selectRows(Array(i)).toDouble(0))
            T_vector = T_vector ++ testing.selectRows(Array(i)).toDouble(0)
        }
        println(T_vector.countPos)
        var T_nonzero = T_vector.countPos

        println(T_vector.countZero)
        T_nonzero
    }


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Construct "selector" matrix I corresponding to training and testing data.
        *  For all values that are not 0 replace with 1 since those are implicit.
        *  @param input  input matrix(training or testing data)
        */
    def ConI(input: MatrixI) : MatrixD = {
        val I = input.copy().toDouble
        for (i <- input.range1; j <- input.range2) {
            if(I(i)(j) > 0)  
                I(i)(j) = 1.0
                
        }
        I
    }

    def rmse(testing: MatrixI, I2: MatrixD, predict: MatrixD): Double = {
        var tar = ((I2 ** (testing.toDouble - predict))**(I2 ** (testing.toDouble - predict))).sum
        println(tar)
        var res = sqrt(tar / getNonZero(testing))
        println(res)
        res
    }

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSRegRecommenderTest` object is used to test the `Recommender` Trait using the MovieLens dataset.
  *  > runMain ALSRegRecommenderTest
  */
object ALSRegRecommenderTest extends App{
    
    val BASE_DIR = System.getProperty("user.dir")
    val data_file =  BASE_DIR + "/data/rating.txt"
    MatrixI.setSp('\t')
    var input   =  MatrixI(data_file)
    input.setCol(0, input.col(0)-1)
    input.setCol(1, input.col(1)-1)
    val (m, n)   = (6040, 3952)
    println("Finished reading m and n.")
    val rec    = new ALSRegRecommender(input, m, n)
    val rating = rec.makeRatings(input, m, n)

    val train_file =  BASE_DIR + "/data/u2Data.train"           // replace u(1-5).base
    val test_file  =  BASE_DIR + "/data/u2Data.test"           // replace u(1-5).test
    println("Imported training and testing set.")
    var train   =  MatrixI(train_file)
    train.setCol(0, train.col(0)-1)
    train.setCol(1, train.col(1)-1)
    println("Generating training set.")
    var tester   =  MatrixI(test_file)
    tester.setCol(0, tester.col(0)-1)
    tester.setCol(1, tester.col(1)-1)
    println("Generating testing set.")
    val train_d = rec.genTrain2(train)
    val testing = rec.genTest2(tester)
    println("Finishing data preprocessing.")

    val I = rec.ConI(train_d)
    val I2 = rec.ConI(testing)

    var predict = new MatrixD(m, n)
    for(i <- 0 until 1) {
        val t = time{
            println("Training Time")
            val t1 = time {                                         // training time
                predict = rec.ALSReg(I, I2)                                      
            } //time
            println("Prediction Time")                              // testing time
            val t2 = time {
                print("RMSE = " + rec.rmse(testing, I2, predict))
            } // time
        } //time
    } // for
}
