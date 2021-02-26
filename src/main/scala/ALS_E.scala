
import scalation.linalgebra.MatrixD
import scalation.linalgebra.Eigenvalue
import scalation.linalgebra.Eigenvector
import scalation.linalgebra._
import scalation.linalgebra.Fac_LU
import scalation.random.RandomMatD
import scalation.random.VariateMat
import scala.util.Random
import scala.math._
import scalation.plot.Plot
import scala.io.Source.fromFile
import scalation.math.double_exp


import MatrixD.eye

class ALS_E(a: MatrixD){
  //Initialize parameters
  var r_lambda = 40 //normalization parameter
  var nf = 200  //dimension of latent vector of each user and item
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 

  //Set up loss function
  //C: confidence matrix
  //P: binary rating matrix
  //X: user latent matrix
  //Y: item latent matrix
  //r_lambda: regularization lambda
  //xTy: predict matrix
  //Total_loss = (confidence_level * predict loss) + regularization loss
  def Lossf( xTy: MatrixD, X: MatrixD, Y: MatrixD, r_lambda: Double): (Double, Double, Double, Double, Double, Double) = {
    val predict_error = (a - xTy) ** (a - xTy)
    val confidence_error = (predict_error).sum
    val mae = ( (a - xTy)).sum / (100000)
    //val mae_round = confidence_error_round / (c.dim1*c.dim2)
    val rmse = sqrt((a - xTy).sum / (100000))
    //val rmse_round = sqrt(confidence_error_round / (c.dim1*c.dim2))
    //println("2.5:"+rmse)
    val regularization =  r_lambda * ((X ** X).sum + (Y ** Y).sum)
    //println("3:"+regularization)
    val total_loss = confidence_error + regularization
    //println("4:"+total_loss)
    
    (predict_error.sum, confidence_error, mae, rmse, regularization , total_loss)
  }

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




  def ALSTrain(): MatrixD = {
    val target = new ALS_E(a)

    var X = RandomMatD (target.nu, target.nf, 1, 0, 1, 0).gen* 0.01
    var Y = RandomMatD (target.ni, target.nf, 1, 0, 1, 0).gen* 0.01


    val interval = 10

    var predict_errors = new VectorD(interval)
    var confidence_errors = new VectorD(interval)
    var regularization_list = new VectorD(interval)
    var total_losses = new VectorD(interval)
    var maes = new VectorD(interval)
    var rmses = new VectorD(interval)
    var total_time_loss = 0.0
    var start_Optimize = 0.0
    var opt_Duration = 0.0

    for (i <- 0 until interval) {
      println("----------------step "+i+"----------------")
      if (i != 0){
          val start_Optimize = System.nanoTime
          target.optimize_user(X, Y, target.nu, target.nf, target.r_lambda)
          target.optimize_item(X, Y, target.ni, target.nf, target.r_lambda)
          val opt_Duration = (System.nanoTime - start_Optimize) / 1e9d
          println{"optimize function running time: "+opt_Duration*1000+" ms"}
      }
      
      val predict = X * (Y.t)

      var start_Loss = System.nanoTime
      val (predict_error, confidence_error, mae, rmse, regularization, total_loss) = target.Lossf(predict, X, Y, target.r_lambda)
      val loss_Duration = (System.nanoTime - start_Loss) / 1e9d

      predict_errors = predict_errors + (i,predict_error)
      confidence_errors = confidence_errors + (i,confidence_error)
      regularization_list = regularization_list + (i, regularization)
      total_losses = total_losses + (i,total_loss)
      maes = maes + (i,mae)
      rmses = rmses + (i,rmse)
      total_time_loss += loss_Duration

      //println("----------------step "+i+"----------------")
      println("predict error: " + predict_error)
      println("confidence error: " + confidence_error)
      println("mae: " + mae)
      //println("mae_round: " + mae_round)
      println("rmse: " + rmse)
      //println("rmse round: " + rmse_round)
      println("regularization: " + regularization)
      println("total loss: " + total_loss)
      println{"loss function running time: "+loss_Duration*1000+" ms"}
    }
  

    var predict = X * (Y.t) 
    predict = predict * (5/predict.max(predict.dim1)) 
    println("----------------final predict----------------")
    val BASE_DIR = System.getProperty("user.dir")
    val data_file =  BASE_DIR + "/data/predict.txt"
    predict.write (data_file)
    //println(predict)
    predict 
  }
}




