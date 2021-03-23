import scalation.linalgebra.MatrixD
import scalation.linalgebra.MatrixI
import scalation.random.RandomMatD
import scala.util.Random
import scala.math.abs
import scala.io.Source.fromFile

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSExplicit` class is used to predict the missing values of an input matrix
 *  by applying Alternating Least Square to factor the matrix.
 *  Once the factors are obtained the missing value in the matrix is obtained as the 
 *  dot product of 'x' and 'y.t', where
 *  <p>
 *      x is the user-factors vector (nf * n)
 *      y is the item-factors vector (nf * m)
 *      predict (i, j) = x dot y.t
 *  <p>
 *------------------------------------------------------------------------------
 *  @param a  the input data matrix
 */
class ALSExplicit(a: MatrixD){

  var r_lambda = 0.065 //normalization parameter
  var nf = 8  //dimension of latent vector of each user and item
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Optimization Function for user. X[u] = (yTy + lambda*I)^-1yTRu
    *  @param X  the user-factors vector
    *  @param Y  the item-factors vector
    *  @param nu  the input matrix's row
    *  @param nf  dimension of latent vector of each user and item
    *  @param r_lambda  the normalization parameter
    */
  def optimize_user(X: MatrixD, Y: MatrixD, nu:Int, nf:Int, r_lambda:Double)={
    val yT = Y.t
    for (i <- X.range1) {
      print(s"\r user: ${i+1} / "+X.dim1)
      val yT_y = yT * Y
      val lI = eye(nf) * r_lambda
      val yT_Ru = ((yT) * ((a.selectRows(Array(i))).t)).col(0)
      X(i) = (yT_y + lI).solve(yT_Ru)
    }
    println()
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Optimization Function for item. Y[i] = (xTx + lambda*I)^-1xTRi
    *  @param Y  the item-factors vector
    *  @param X  the user-factors vector
    *  @param ni  the input matrix's column
    *  @param nf  dimension of latent vector of each user and item
    *  @param r_lambda  the normalization parameter
    */
  def optimize_item(X: MatrixD, Y: MatrixD, ni:Int, nf:Int, r_lambda:Double) = {
    val xT = X.t
    for (i <- Y.range1) {
      print(s"\r item: ${i+1} / "+Y.dim1)
      val xT_x = xT * X
      val lI = eye(nf) * r_lambda
      val xT_Ri = ((xT) * (a.selectCols(Array(i)))).col(0)
      Y(i) = (xT_x + lI).solve(xT_Ri)
    }
    println()
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Get the average value of the input dataset for normalization
    *  @param file_name  the input dataset's path
    */
  def getAverage(file_name: String): Double = {
    val tar = MatrixI(file_name)
    tar.col(2).sum / tar.dim1.toDouble
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Train the model to get predict matrix using input dataset.
    *  @see Matrix Completion via Alternating Least Square(ALS)
    *  @see http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf 
    */
  def train(): MatrixD = {
    val target = new ALSExplicit(a)
    val BASE_DIR = System.getProperty("user.dir")
    val www =  BASE_DIR + "/data/a.txt"
    a.write(www)
    var X = RandomMatD (target.nu, target.nf, 5, 1, 1, 0).gen
    var Y = RandomMatD (target.ni, target.nf, 5, 1, 1, 0).gen
    
    //var X = RandomMatD (target.nu, target.nf, 5, 1, 1, 0).gen* 0.00001
    //var Y = RandomMatD (target.ni, target.nf, 5, 1, 1, 0).gen* 0.00001
    //print("X = " + X)
    //print("Y = " + Y)
    val interval = 10 //iterations

    for (i <- 0 until interval) {
      println("----------------step "+i+"----------------")
      if (i != 0){
          target.optimize_user(X, Y, target.nu, target.nf, target.r_lambda)
          target.optimize_item(X, Y, target.ni, target.nf, target.r_lambda)
      }//compute X and Y iteratively
      
      val predict = X * (Y.t) 

    }


    //val BASE_DIR = System.getProperty("user.dir")
    val test_file =  BASE_DIR + "/data/u2Data.test"
    val ave = getAverage(test_file)
    var predict = X * (Y.t) + ave -1

    println("----------------End----------------")
    predict 
  }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSExplicitTest` companion object is used to perform imputation.
 *  This part is under development, because here we need implicit data, so r here
 *  needs data like playing hours(ia will be matrix contains 0-1 here)
 */
object ALSExplicitTest extends App{
   val r = new MatrixD ((10, 11), 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0,
                                 0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0,
                                 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0,
                                 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5,
                                 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4,
                                 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0,
                                 0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0
                                 ) //input matrix
    println (s"r = $r")

    val als = new ALSExplicit(r)
    val ia = als.train()  //generate predict matrix
    println (s"ia = $ia")
    println ("Predicted value for(5, 6) = " + ia(5, 6))

}







