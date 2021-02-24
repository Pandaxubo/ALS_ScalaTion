
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


class CFRecommender (input: MatrixI, m: Int, n: Int) extends Recommender{
    private var predicted  = new MatrixD(m, n)                                // Matrix for storing SVD predicted values
    private val ratings = makeRatings(input, m, n)                              // original ratings matrix
    private var training = new MatrixD(ratings.dim1, ratings.dim2)              // training dataset
    private var copy_training = new MatrixD(ratings.dim1, ratings.dim2)         // copy of training dataset
    private val als = new ALS(training)
    var conc = als.ConC(predicted)
    


    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a rating based on Singular value Decompostion of matrix
      *  @param i  user
      *  @param j item
      */
    def rate (i: Int, j: Int) : Double = predicted(i, j)


      //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Generates the training matrix for the dataset for Phase 1
      *  @param train : training data matrix
      */
    def genTrain2 (train: MatrixI)
    {
        for (i <- train.range1) training(train(i, 0), train(i, 1)) = train(i, 2)
        copy_training = training.copy()
    } // genTrain2


    def ALS{
        predicted = als.ALSTrain()
    }


    def ConTestP(a: MatrixI) : MatrixI = {
        val P = a.copy()
            for (i <- a.range1 ) {
                for (j <- a.range2 ) {
            if(P(i,j) > 0)  
            P(i,j) = 1
                }
            }
        P
    }


    def error_metric (input: MatrixI)
    {   
        conc = makeRatings(input, m, n)* 40 + 1
        var conp = ConTestP(input)
        var sum1, sum2, sum3, sum4 = 0.0
        for (i <- input.range1) {
            //val a = input(i, 2).toDouble
            val a = conp (input(i, 0), input(i, 1))
            val c = conc (input(i, 0), input(i, 1))
            val p = rate (input(i, 0), input(i, 1))
            
            if (! p.isNaN) {
                sum1 += c * abs(a - p)                               // non rounded MAE
                sum2 += c * abs (a - round (p))                       // rounded MAE
                sum3 += c * (a - p) * (a - p)                         // non rounded RMSE
                sum4 += c * (a - round (p)) * (a - round (p))         // rounded RMSE
            } // if
        } //for
        println ("MAE  Non-Rounded = " + sum1 / input.dim1)
        println ("MAE      Rounded = " + sum2 / input.dim1)
        println ("RMSE Non-Rounded = " + sqrt (sum3 / input.dim1))
        println ("RMSE     Rounded = " + sqrt (sum4 / input.dim1))
    } // error metrics
}

object CFRecommenderTest extends App{

    
    val BASE_DIR = System.getProperty("user.dir")
    val data_file =  BASE_DIR + "/data/sorted_data.txt"
    MatrixI.setSp('\t')
    var input   =  MatrixI(data_file)
    input.setCol(0, input.col(0)-1)
    input.setCol(1, input.col(1)-1)
    val (m, n)   = (943, 1682)
    val rec    = new CFRecommender(input, m, n)
    val rating = rec.makeRatings(input, m, n)
    //println(rating.col(1))

    val train_file =  BASE_DIR + "/data/u2.base"           // replace u(1-5).base
    val test_file  =  BASE_DIR + "/data/u2.test"           // replace u(1-5).test

    var train   =  MatrixI(train_file)
    train.setCol(0, train.col(0)-1)
    train.setCol(1, train.col(1)-1)

    var tester   =  MatrixI(test_file)
    tester.setCol(0, tester.col(0)-1)
    tester.setCol(1, tester.col(1)-1)

    rec.genTrain2(train)


    for(i <- 0 until 1) {
        val t = time{
            println("Training Time")
            val t1 = time {                                         // training time
                rec.ALS                                       // Pure SVD
                //recMB.SVDR                                          // regualarized SVD
            } //time
            println("Prediction Time")                              // testing time
            val t2 = time {
                rec.error_metric(tester)
            } // time
        } //time
    } // for
}