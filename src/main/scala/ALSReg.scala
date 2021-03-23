import scalation.linalgebra.MatrixD
import scalation.linalgebra.VectorD
import scalation.linalgebra.MatrixI
import scalation.random.RandomMatD
import scala.util.Random
import scala.math.abs
import scala.io.Source.fromFile

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ALSReg` class is used to predict the missing values of an input matrix
 *  by applying Alternating Least Square to factor the matrix. Compare to ALS Explicit
 *  method, it will be more accurate and faster.
 *  Once the factors are obtained the missing value in the matrix is obtained as the 
 *  dot product of 'x.t' and 'y', where
 *  <p>
 *      x is the user-factors vector (nf * n)
 *      y is the item-factors vector (nf * m)
 *      predict (i, j) = x.t dot y
 *  <p>
 *------------------------------------------------------------------------------
 *  @param a  the input data matrix
 */
class ALSReg(a: MatrixD){

  var r_lambda = 0.1 //normalization parameter
  var nf = 20  //dimension of latent vector of each user and item
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 
  var n_epochs = 15   //Number of epochs
  var E = eye(nf) //(nf x nf)-dimensional idendity matrix

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Optimization Function for user. 
    *  @param X  the user-factors vector
    *  @param Y  the item-factors vector
    *  @param E  the compressed matrix(nf * nf)
    *  @param I  "selector" matrix I 
    *  @param r_lambda  the normalization parameter
    */
  def optimize_user(X: MatrixD, Y: MatrixD, r_lambda:Double, E: MatrixD, I: MatrixD)={
    for (i <- X.range2) {
      print(s"\r user: ${i+1} / "+ X.dim2)

      var nui = 0
      var Ii_nonzero = new VectorD(1)
      for(j <- 0 to I.dim2 - 1){
          if(I(i)(j) != 0){
              if (j == 0){
                Ii_nonzero(0) = j
                nui += 1
              }
              else{
                nui += 1
                Ii_nonzero = Ii_nonzero ++ j
              }
          }
      }
      //println(Ii_nonzero)
      
      if(nui == 0)  nui = 1

      var Y_Ii = new MatrixD(Y.dim1, Ii_nonzero.size)
      for(j <- Y.range1; k <- 0 until Ii_nonzero.size){//; value <- Ii_nonzero){
        Y_Ii(j)(k) =  Y(j)(:Ii_nonzero)
      }

      var a_Ii = new MatrixD(Ii_nonzero.size, 1)
      for(j <- 0 until Ii_nonzero.size; value <- Ii_nonzero){  
        a_Ii(j)(0) =  a(i)(value.toInt)
      }

      var Ai = Y_Ii * Y_Ii.t + E * r_lambda * nui
      var Vi = (Y_Ii * a_Ii).col(0)

      X.setCol(i, (Ai).solve(Vi))

    }
    println()
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**  Optimization Function for item.
    *  @param X  the user-factors vector
    *  @param Y  the item-factors vector
    *  @param E  the compressed matrix(nf * nf)
    *  @param I  "selector" matrix I 
    *  @param r_lambda  the normalization parameter
    */
  def optimize_item(X: MatrixD, Y: MatrixD, r_lambda:Double, E: MatrixD, I: MatrixD) = {
    for (j <- Y.range2) {
      print(s"\r item: ${j+1} / "+ Y.dim2)

      var nmj = 0
      var Ij_nonzero = new VectorD(1)
      for(i <- I.range2){
          if(I(j)(i) != 0){
            if (j == 0){
              Ij_nonzero(0) = i
              nmj += 1
              }
            else{
              nmj += 1
              Ij_nonzero = Ij_nonzero ++ i
            }
          }
      }
      if(nmj == 0)  nmj = 1

      var X_Ij = new MatrixD(X.dim1, Ij_nonzero.size)
      for(i <- X.range1; k <- 0 until Ij_nonzero.size; value <- Ij_nonzero){
        X_Ij(i)(k) =  X(i)(value.toInt)
      }

      var a_Ij = new MatrixD(Ij_nonzero.size, 1)
      for(i <- 0 until Ij_nonzero.size; value <- Ij_nonzero){        
        a_Ij(i)(0) =  a(j)(value.toInt)
      }

      var Aj = X_Ij * X_Ij.t + E * r_lambda * nmj
      var Vj = (X_Ij * a_Ij).col(0)
      
      Y.setCol(j, (Aj).solve(Vj))
    }
    println()
  }

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Train the model to get predict matrix using input dataset.
    *  @see Large-scale Parallel Collaborative Filtering for the Netflix Prize
    *  @see http://shiftleft.com/mirrors/www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf 
    */
  def train(I: MatrixD, I2: MatrixD): MatrixD = {
    val target = new ALSReg(a)
    println(I.dim1+","+I.dim2)
    var X = RandomMatD (target.nf, target.nu, 1, 0, 1, 0).gen * 3
    var Y = RandomMatD (target.nf, target.ni, 1, 0, 1, 0).gen * 3

    var Y_vector: VectorD = Y(0)
    for(i <- 1 until Y.dim1){
        Y_vector ++ Y(i)
    }
    var avgRating = Y_vector.sum/(Y_vector.size - Y_vector.countZero)
    for(i <- Y.range2){
      Y(0)(i) = avgRating
    }


    for (i <- 0 until n_epochs) {
      println("----------------step "+i+"----------------")
      if (i != 0){
          target.optimize_user(X, Y, target.r_lambda, target.E, I) // may not always be I
          target.optimize_item(X, Y, target.r_lambda, target.E, I)
      }//compute X and Y iteratively
    }

    var predict = (X.t) * Y

    println("----------------End----------------")
    predict
  }
}






