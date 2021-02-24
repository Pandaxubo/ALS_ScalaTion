
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



import MatrixD.eye

class ALS(a: MatrixD){
  //Initialize parameters
  var r_lambda = 40 //normalization parameter
  var nf = 200  //dimension of latent vector of each user and item
  val alpha = 40  //confidence level
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 

  // Initialize Binary Rating Matrix P
  def ConP(nu : Int, ni : Int) : MatrixD = {
    val P = a.copy()
		for (i <- a.range1 ) {
			for (j <- a.range2 ) {
        if(P(i,j) > 0)  
          P(i,j) = 1.0
			}
		}
    //println(P.toString())
    P
  }

  // Initialize Confidence Matrix C
  def ConC(a: MatrixD) : MatrixD = a * alpha + 1

  //Set up loss function
  //C: confidence matrix
  //P: binary rating matrix
  //X: user latent matrix
  //Y: item latent matrix
  //r_lambda: regularization lambda
  //xTy: predict matrix
  //Total_loss = (confidence_level * predict loss) + regularization loss
  def Lossf(c: MatrixD, p: MatrixD, xTy: MatrixD, X: MatrixD, Y: MatrixD, r_lambda: Int): (Double, Double, Double, Double, Double, Double) = {
    val predict_error = (p - xTy) ** (p - xTy)
    //val predict_error_round = square(xTy - round(p))
    //println("1:"+predict_error)
    val confidence_error = (c**predict_error).sum
    //val confidence_error_round = (c**predict_error_round).sum
    //println("2:"+confidence_error)
    val mae = (c ** (p - xTy)).sum / (100000)
    //val mae_round = confidence_error_round / (c.dim1*c.dim2)
    val rmse = sqrt((c ** (p - xTy)).sum / (100000))
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

  def ALSTrain(): MatrixD = {
    val target = new ALS(a)

    var X = RandomMatD (target.nu, target.nf, 1, 0, 1, 0).gen* 0.01
    var Y = RandomMatD (target.ni, target.nf, 1, 0, 1, 0).gen* 0.01
  
    val p = target.ConP(target.nu,target.ni)
    val c = target.ConC(a)

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
          target.optimize_user(X, Y, c, p, target.nu, target.nf, target.r_lambda)
          target.optimize_item(X, Y, c, p, target.ni, target.nf, target.r_lambda)
          val opt_Duration = (System.nanoTime - start_Optimize) / 1e9d
          println{"optimize function running time: "+opt_Duration*1000+" ms"}
      }
      
      val predict = X * (Y.t)

      var start_Loss = System.nanoTime
      val (predict_error, confidence_error, mae, rmse, regularization, total_loss) = target.Lossf(c, p, predict, X, Y, target.r_lambda)
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
    println("----------------final predict----------------")
    
    //println(predict)
    predict 
  }
}


object ALSTest extends App{
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


  def read(fileName: String): MatrixD = {
    var lines = fromFile(fileName).getLines.toArray        // get the lines from file
    val (m, n) = (lines.length, lines(0).split("\t").length)
    val x = new MatrixD(m, n)
    for (i <- x.range1) {
      x(i) = VectorD (lines(i).split ("\t"))    
    } // for
    x
  } // apply
  
  val start = System.nanoTime
  // val BASE_DIR = System.getProperty("user.dir")
  // val data_file =  BASE_DIR + "/src/main/scala/new_SortedM.txt"
  // //val x = new ALS()
  // var r = read(data_file)
  // // r.setCol(0, (r.col(0)-1))
  // // r.setCol(1, (r.col(1)-1))
  val target = new ALS(r)
  
  var X = RandomMatD (target.nu, target.nf, 1, 0, 1, 0).gen* 0.01
  var Y = RandomMatD (target.ni, target.nf, 1, 0, 1, 0).gen* 0.01
  

  val p = target.ConP(target.nu,target.ni)
  val c = target.ConC(r)
  

  var interval = 10


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
        target.optimize_user(X, Y, c, p, target.nu, target.nf, target.r_lambda)
        target.optimize_item(X, Y, c, p, target.ni, target.nf, target.r_lambda)
        val opt_Duration = (System.nanoTime - start_Optimize) / 1e9d
        println{"optimize function running time: "+opt_Duration*1000+" ms"}
    }
    
    val predict = X * (Y.t)

    var start_Loss = System.nanoTime
    val (predict_error, confidence_error, mae, rmse, regularization, total_loss) = target.Lossf(c, p, predict, X, Y, target.r_lambda)
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
  println("----------------final predict----------------")
  
  //println(predict)

  val program_run = (System.nanoTime - start) / 1e9d
  println{"Total loss function running time: "+total_time_loss*1000+" ms"}
  println{"Program Running time: "+program_run*1000+" ms"}
  
  // for(line <- fromFile(data_file).getLines.toArray){
  //   println(line)
  //   println(line.split("\t").length)
  // }

  val t_idx = VectorD.range(0, interval)
  new Plot (t_idx, predict_errors, null, "P_ERROR", true)
  new Plot (t_idx, confidence_errors, null, "C_ERROR", true)
  new Plot (t_idx, maes, null, "MAE", true)
  new Plot (t_idx, rmses, null, "RMSE", true)
  new Plot (t_idx, regularization_list, null, "R_LIST", true)
  new Plot (t_idx, total_losses, null, "TOTAL_LOSS", true)
  //println(fromFile(data_file).getLines.split(" "))
  //println{r}
}




