import scalation.linalgebra.MatrixD
import scalation.linalgebra.MatrixI
import scalation.random.RandomMatD
import scala.util.Random
import scala.math.abs

import MatrixD.eye

class ALSExplicit(a: MatrixD){
  //Initialize parameters
  var r_lambda = 1 //normalization parameter
  var nf = 10  //dimension of latent vector of each user and item
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 

  //Optimization Function for user and item
  //X[u] = (yTCuy + lambda*I)^-1yTCuy
  //Y[i] = (xTCix + lambda*I)^-1xTCix
  //two formula is the same when it changes X to Y and u to i
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

  def getAverage(file_name: String): Double = {
    val tar = MatrixI(file_name)
    tar.col(2).sum / tar.dim1.toDouble
  }

  def train(): MatrixD = {
    val target = new ALSExplicit(a)

    var X = RandomMatD (target.nu, target.nf, 5, 1, 1, 0).gen* 0.00001
    var Y = RandomMatD (target.ni, target.nf, 5, 1, 1, 0).gen* 0.00001

    val interval = 10

    for (i <- 0 until interval) {
      println("----------------step "+i+"----------------")
      if (i != 0){
          target.optimize_user(X, Y, target.nu, target.nf, target.r_lambda)
          target.optimize_item(X, Y, target.ni, target.nf, target.r_lambda)
      }
      
      val predict = X * (Y.t) 

    }


    val BASE_DIR = System.getProperty("user.dir")
    val test_file =  BASE_DIR + "/data/u2.test"
    val ave = getAverage(test_file)
    var predict = X * (Y.t)  + ave -1

    println("----------------End----------------")
    predict 
  }
}

//under development
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
                                 ) 
    println (s"r = $r")

    val als = new ALSExplicit(r)
    val ia = als.train()
    println (s"ia = $ia")
    println ("Predicted value for(5, 6) = " + ia(5, 6))
}







