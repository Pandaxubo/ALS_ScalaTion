
import scalation.linalgebra.MatrixD
import scalation.linalgebra.Eigenvalue
import scalation.linalgebra.Eigenvector
import scalation.linalgebra._
import scalation.linalgebra.Fac_LU
import scala.util.Random
import scala.math._
import scalation.plot.Plot
import scala.io.Source.fromFile



import MatrixD.eye

class MDuse(a: MatrixD){
  //Initialize parameters
  var r_lambda = 40 //normalization parameter
  var nf = 200  //dimension of latent vector of each user and item
  val alpha = 40  //confidence level
  val ni = a.dim2 //number of items 
  val nu = a.dim1 //number of users 

  //created a matrix with random values
  def createRM(row_number : Int, column_number : Int) : MatrixD = {
 	  var matrixA = new MatrixD (row_number, column_number)
    var i = 0
    var j = 0
		for (i <- 0 to row_number-1 ) {
			for (j <- 0 to column_number-1 ) {
				matrixA(i,j) = Random.nextInt(10000) * 0.000001
        
			}
		}
	  matrixA
	} // create a random matrix 

  //Replace all elements in the matrix with squared value
  def square(a:MatrixD): (MatrixD) = {
    val S = a.copy()
    for (i <- 0 to a.dim1-1 ) {
			for (j <- 0 to a.dim2-1 ) { 
          S(i,j) = a(i,j) * a(i,j)
			}
		}
    S
  }

  def round(a:MatrixD): (MatrixD) = {
    val S = a.copy()
    for (i <- 0 to a.dim1-1 ) {
			for (j <- 0 to a.dim2-1 ) {
        val temp = a(i,j)
        S(i,j) = temp.round
			}
		}
    S
  }

  // Initialize Binary Rating Matrix P
  def ConP(nu : Int, ni : Int) : MatrixD = {
    val P = a.copy()
		for (i <- 0 to a.dim1-1 ) {
			for (j <- 0 to a.dim2-1 ) {
        if(P(i,j) > 0)  
          P(i,j) = 1.0
			}
		}
    //println(P.toString())
    P
  }

  // Initialize Confidence Matrix C
  def ConfC(a: MatrixD) : MatrixD = {
    val C = a * alpha + 1
    C
  }

  //Set up loss function
  //C: confidence matrix
  //P: binary rating matrix
  //X: user latent matrix
  //Y: item latent matrix
  //r_lambda: regularization lambda
  //xTy: predict matrix
  //Total_loss = (confidence_level * predict loss) + regularization loss
  def Lossf(c: MatrixD, p: MatrixD, xTy: MatrixD, X: MatrixD, Y: MatrixD, r_lambda: Int): (Double, Double, Double, Double, Double, Double) = {
    val predict_error = square(p - xTy)
    //val predict_error_round = square(xTy - round(p))
    //println("1:"+predict_error)
    val confidence_error = (c**predict_error).sum
    //val confidence_error_round = (c**predict_error_round).sum
    //println("2:"+confidence_error)
    val mae = confidence_error / (c.dim1*c.dim2)
    //val mae_round = confidence_error_round / (c.dim1*c.dim2)
    val rmse = sqrt(confidence_error / (c.dim1*c.dim2))
    //val rmse_round = sqrt(confidence_error_round / (c.dim1*c.dim2))
    //println("2.5:"+rmse)
    val regularization =  r_lambda * (square(X).sum + square(Y).sum)
    //println("3:"+regularization)
    val total_loss = confidence_error + regularization
    //println("4:"+total_loss)
    
    (predict_error.sum, confidence_error, mae, rmse, regularization , total_loss)
  }

  //construct diagonal matrix with row
  def diagMR(mat : MatrixD, tarRow: Int) : MatrixD = {
    val tempMat = mat.selectRows(Array(tarRow))
    val tarMat = new MatrixD(tempMat.dim2, tempMat.dim2)
    for(i <- 0 to tempMat.dim2 -1){
        tarMat(i,i) = tempMat(0,i)
    }
    tarMat
  }
 
  //construct diagonal matrix with column
  def diagMC(mat : MatrixD, tarCol: Int) : MatrixD = {
    val tempMat = mat.selectCols(Array(tarCol))
    
    val tarMat = new MatrixD(tempMat.dim1, tempMat.dim1)
    for(i <- 0 to tarMat.dim1-1){
        tarMat(i,i) = tempMat(i,0)
        //println(tempMat)
        //println(tarMat)
    }
    tarMat
  }

  //set row to the matrix row
  def setRow(mat : MatrixD, tarRow: VectoD, index: Int) : MatrixD = {
    
    for(j <- 0 to mat.dim2 -1){
        mat(index,j) = tarRow(j)
    }
    mat
  }

  //set column to the matrix row
  def setCol(mat : MatrixD, tarCol: VectoD, index: Int) : MatrixD = {
    
    for(j <- 0 to mat.dim2 -1){
        mat(j,index) = tarCol(j)
    }
    mat
  }

  //convert vector to matrix
  def makeVec(mat : MatrixD) : VectoD = {
    val vec = new VectorD(mat.dim1)
    
    for(j <- 0 to mat.dim1 -1){
        vec(j) = mat(j,0)
    }
    vec
  }
  

  //Optimization Function for user and item
  //X[u] = (yTCuy + lambda*I)^-1yTCuy
  //Y[i] = (xTCix + lambda*I)^-1xTCix
  //two formula is the same when it changes X to Y and u to i
  def optimize_user(X: MatrixD, Y: MatrixD, C: MatrixD, P: MatrixD, nu:Int, nf:Int, r_lambda:Int)={
    val yT = Y.t
    
    for (i <- 0 to nu-1 ) {
      print(s"\r user: ${i+1} / "+nu)
      val Cu = diagMR(C, i)
      
      val yT_Cu_y = yT * Cu * Y
      val lI = eye(nf) * r_lambda
      // val yT_Cu_pu = makeVec((yT * Cu) * ((P.selectRows(Array(i))).t))
      val yT_Cu_pu = ((yT * Cu) * ((P.selectRows(Array(i))).t)).col(0)
      
      setRow(X, solveM(yT_Cu_y + lI, yT_Cu_pu), i)
    }
    println()
  }

  def optimize_item(X: MatrixD, Y: MatrixD, C: MatrixD, P: MatrixD, ni:Int, nf:Int, r_lambda:Int) = {
    val xT = X.t

    for (i <- 0 to ni-1 ) {
      print(s"\r item: ${i+1} / "+(ni))
      val Ci = diagMC(C, i)
      val xT_Ci_x = xT * Ci * X
      val lI = eye(nf) * r_lambda
      //val xT_Ci_pi = makeVec((xT * Ci) * (P.selectCols(Array(i))))
      val xT_Ci_pi = ((xT * Ci) * (P.selectCols(Array(i)))).col(0)
      //var aaa = xT_Ci_x + lI
      setRow(Y, solveM(xT_Ci_x + lI, xT_Ci_pi), i)
      //Y.setCol(i, solveM(xT_Ci_x + lI, xT_Ci_pi))
    }
    println()
  }
  
  //solve lu decomposition problem
  def solveM(a: MatrixD , b: VectoD): VectoD = {
      a.solve(lud_npp_new(a), b)
    }

     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor 'this' matrix into the product of upper and lower triangular
     *  matrices '(l, u)' using the 'LU' Factorization algorithm.(Update version to Scalation 1.7)
     *  Caveat:  This version requires square matrices and performs no partial pivoting.
     *  @see `Fac_LU` for a more complete implementation
     */
  def lud_npp_new(tar : MatrixD): (MatrixD, MatrixD) ={
        
    val l = new MatrixD (tar.dim1)          // lower triangular matrix
    val u = new MatrixD (tar)          // upper triangular matrix (a copy of this)  
    for(i <- 0 to tar.dim1 - 1) {
      var pivot = u(i, i)
      l(i,i) = 1.0   
      for(j <- i + 1 until tar.dim2){
       l(i, j) = 0.0
      }      
      for(k <- i + 1 until tar.dim1) { 
        var mult = u(k, i) / pivot
        l(k, i) = mult
        for(j <- 0 to tar.dim2 - 1) {  
          u(k, j) = u(k ,j) -  mult * u(i ,j)
        }  
      }
    }  
    (l, u)
  }
}


object MDTest extends App{
  //  val r = new MatrixD ((10, 11), 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0,
  //                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
  //                                0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0,
  //                                0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0,
  //                                0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0,
  //                                0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0,
  //                                0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5,
  //                                0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4,
  //                                0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0,
  //                                0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0
  //                                )
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
  val BASE_DIR = System.getProperty("user.dir")
  val data_file =  BASE_DIR + "/src/main/scala/new_SortedM.txt"
  //val x = new MDuse()
  var r = read(data_file)
  // r.setCol(0, (r.col(0)-1))
  // r.setCol(1, (r.col(1)-1))
  val target = new MDuse(r)
  
  var X = target.createRM(target.nu, target.nf)
  var Y = target.createRM(target.ni, target.nf)
  

  val p = target.ConP(target.nu,target.ni)
  val c = target.ConfC(r)
  

  var interval = 15


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




