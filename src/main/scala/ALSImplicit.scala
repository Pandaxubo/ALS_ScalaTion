import scalation.linalgebra.MatrixD
import scalation.random.RandomMatD
import scala.util.Random
import scala.math.abs

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSImplicit` class is used to predict the missing values of an input matrix
 *  which is implicit data by applying Alternating Least Square to factor the matrix.
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
class ALSImplicit(a: MatrixD){

  var r_lambda = 40 //normalization parameter
  var nf = 10  //dimension of latent vector of each user and item
  val alpha = 150  //confidence level
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 


  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Initialize Binary Rating Matrix P of the input matrix. For all values that are
    *  not 0 replace with 1 since those are implicit.
    *  @param nu  the input matrix's row
    *  @param ni  the input matrix's column
    */
  // 
  def ConP(nu : Int, ni : Int) : MatrixD = {
    val P = a.copy()
		for (i <- a.range1 ) {
			for (j <- a.range2 ) {
        if(P(i,j) > 0)  
          P(i,j) = 1.0
			}
		}
    P
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Initialize Confidence Matrix C to measure our confidence in observing pui. 
    *  Alpha controls the rate of increase.
    *  @param a  the input matrix
    */
  // 
  def ConC(a: MatrixD) : MatrixD = a * alpha + 1



  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Optimization Function for user. X[u] = (yTCuy + lambda*I)^-1yTCuy
    *  @param X  the user-factors vector
    *  @param Y  the item-factors vector
    *  @param C  the confidence matrix
    *  @param P  the binary rating matrix
    *  @param nu  the input matrix's row
    *  @param nf  dimension of latent vector of each user and item
    *  @param r_lambda  the normalization parameter
    */
  def optimize_user(X: MatrixD, Y: MatrixD, C: MatrixD, P: MatrixD, nu:Int, nf:Int, r_lambda:Int)={
    val yT = Y.t
    for (i <- X.range1) {
      print(s"\r user: ${i+1} / "+X.dim1)
      //val Cu = diagMR(C, i)
      val Cu = eye(C.dim2) ** C(i)
      //val Cu = C.getDiag().apply
      val yT_Cu_y = yT * Cu * Y
      val lI = eye(nf) * r_lambda
      val yT_Cu_pu = ((yT * Cu) * ((P.selectRows(Array(i))).t)).col(0)
      X(i) = (yT_Cu_y + lI).solve(yT_Cu_pu)
    }
    println()
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Optimization Function for item. Y[i] = (xTCix + lambda*I)^-1xTCix
    *  @param Y  the item-factors vector
    *  @param X  the user-factors vector
    *  @param C  the confidence matrix
    *  @param P  the binary rating matrix
    *  @param ni  the input matrix's column
    *  @param nf  dimension of latent vector of each user and item
    *  @param r_lambda  the normalization parameter
    */
  def optimize_item(X: MatrixD, Y: MatrixD, C: MatrixD, P: MatrixD, ni:Int, nf:Int, r_lambda:Int) = {
    val xT = X.t
    for (i <- Y.range1) {
      print(s"\r item: ${i+1} / "+Y.dim1)
      //val Ci = diagMC(C, i)
      val Ci = eye(C.dim1) ** C.col(i)
      val xT_Ci_x = xT * Ci * X
      val lI = eye(nf) * r_lambda
      val xT_Ci_pi = ((xT * Ci) * (P.selectCols(Array(i)))).col(0)
      Y(i) = (xT_Ci_x + lI).solve(xT_Ci_pi)
    }
    println()
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Train the model to get predict matrix using input dataset.
    *  @see Collaborative Filtering for Implicit Feedback Datasets
    *  @see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781121
    */
  def train(): MatrixD = {
    val target = new ALSImplicit(a)

    var X = RandomMatD (target.nu, target.nf, 1, 0, 1, 0).gen* 0.01
    var Y = RandomMatD (target.ni, target.nf, 1, 0, 1, 0).gen* 0.01
  
    val p = target.ConP(target.nu,target.ni)
    val c = target.ConC(a)

    val interval = 10 //iterations

    for (i <- 0 until interval) {
      println("----------------step "+i+"----------------")
      if (i != 0){
          target.optimize_user(X, Y, c, p, target.nu, target.nf, target.r_lambda)
          target.optimize_item(X, Y, c, p, target.ni, target.nf, target.r_lambda)
      } //compute X and Y iteratively
      
      val predict = X * (Y.t)
    }
    
    var predict = X * (Y.t) //final predict matrix
    println("----------------End----------------")
    
    //normalization
    predict = (predict + abs(predict.min(predict.range1,predict.range2)))
    predict = predict*(1/predict.max(predict.range1,predict.range2))
    
    predict 
  }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSImplicitTest` companion object is used to perform imputation.
 *  This part is under development, because here we need implicit data, so r here
 *  needs data like playing hours(ia will be matrix contains 0-1 here)
 */
object ALSImplicitTest extends App{
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
                                 ) //training matrix
  val r_test = new MatrixD ((10, 11), 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                                 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
                                 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                 0, 0, 0, 1, 0, 0, 0, 0, 4, 1, 0
                                 ) //testing matrix
    println (s"r = $r")

    val als = new ALSImplicit(r)
    val ia = als.train()  //generate predict matrix
    println (s"ia = $ia")
    println ("Predicted value for(5, 6) = " + ia(5, 6))
}




