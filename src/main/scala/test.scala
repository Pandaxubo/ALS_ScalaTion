

import scalation.linalgebra.MatrixD
import scalation.linalgebra.Eigenvalue
import scalation.linalgebra.Eigenvector
import scalation.linalgebra._
import scalation.linalgebra.Fac_LU
import scala.util.Random
import scalation.linalgebra.VectorD

import MatrixD.eye


class AirYan(a: MatrixD){
  def solveM(a: MatrixD , b: VectoD): VectoD = {
      a.solve(lud_npp_new(a), b)
    }

     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor 'this' matrix into the product of upper and lower triangular
     *  matrices '(l, u)' using the 'LU' Factorization algorithm.
     *  Caveat:  This version requires square matrices and performs no partial pivoting.
     *  @see `Fac_LU` for a more complete implementation
     */
    def lud_npp_new(tar : MatrixD): (MatrixD, MatrixD) =
    {
        
    //     val l = new MatrixD (tar.dim1)          // lower triangular matrix
    //     val u = new MatrixD (tar.copy)          // upper triangular matrix (a copy of this)  

    //     for(j <- 0 to tar.dim2 ) {             
    //         for(i <- j+1 to tar.dim1 -1 ) {  
    //             var mult = tar(i, j) / tar(j, j);  
    //             println("mult ="+ mult) 
    //             for(k <- j to tar.dim2-1) {  
    //                  u(i, k) = tar(i ,k) - tar(j ,k ) * mult
    //                  println("u : i = "+i + ", j = "+j) 
    //                  //得出上三角矩阵U,通过减去矩阵的第一行,第二行,第一行(第二行)得到上三角矩阵
    //              }  
    //             l(i, j) = mult  //得到下三角矩阵是得出上三角矩阵的乘积因子
    //             println("l : i = "+i + ", j = "+j)
    //         }  
    //     }  
    //     println(l.toString())
    //     println(u)
    //     println(l * u)
    //     (l, u)
    // }

        var l = new MatrixD (tar.dim1)          // lower triangular matrix
        var u = new MatrixD (tar)          // upper triangular matrix (a copy of this)  
        println(u)
        for(i <- 0 to tar.dim1 - 1) {
            var pivot = u(i, i)
            
            println("pivot = " + u(i, i))
            l(i,i) = 1.0   
            for(j <- i + 1 until tar.dim2){
                l(i, j) = 0.0
            }      
            println("before lu : ")
            println(l.toString())
            println(u)
            for(k <- i + 1 until tar.dim1) { 
                var mult = u(k, i) / pivot
                
                l(k, i) = mult
                println("mult ="+ mult, "asdfsd:"+l(k,i)) 
                for(j <- 0 to tar.dim2 - 1) {  
                     u(k, j) = u(k ,j) -  mult * u(i ,j)
                    
                     println("u : i = "+i + ", j = "+j+",u(k, j):"+ u(k, j) ) 
                     //得出上三角矩阵U,通过减去矩阵的第一行,第二行,第一行(第二行)得到上三角矩阵
                 }  
                  //得到下三角矩阵是得出上三角矩阵的乘积因子
                println("l : i = "+i + ", k = "+k)
            }
            //println("uuuuuuuuu", u)  
        } 


        println(l)
        println(u)
        println(l * u)
        (l, u)
    }




//     def lud_n(mat: MatrixD, n: Int) : (MatrixD, MatrixD) = {
//       var lower = new MatrixD (n )          // lower triangular matrix
//       var upper = new MatrixD (n)          // upper triangular matrix (a copy of this)


//     // Decomposing matrix into Upper and Lower
// 		// triangular matrix
// 		for (i <- 0 until n)
// 		{
// 			// Upper Triangular
// 			for (k <- i until n) 
// 			{
// 				// Summation of L(i, j) * U(j, k)
// 				var sum = 0
// 				for (j <- 0 until i)
// 					sum += (lower(i, j) * upper(j, k))

// 				// Evaluating U(i, k)
// 				upper(i, k) = mat(i, k) - sum;
// 			}

// 			// Lower Triangular
// 			for (k <- i until n) 
// 			{
// 				if (i == k)
// 					lower(i ,i ) = 1; // Diagonal as 1
// 				else
// 				{
// 					// Summation of L(k, j) * U(j, i)
// 					var sum = 0
// 					for (j <- 0 until i)
// 						sum += (lower( k ,j) * upper(j , i));

// 					// Evaluating L(k, i)
// 					lower(k, i) = (mat(k,i ) - sum) / upper(i,i);
// 				}
//     }
//     println(upper);
//     println(lower);
// 		(lower, upper)
//   }
// }
}

object AirTest extends App{
  // val r = new MatrixD((4,4), 2, 4, 1, -3,
  //                           -1, -2, 2, 4,
  //                            4, 2, -3, 5, 
  //                            5, -4, -3, 1
  //  )
  val r = new MatrixD((3,3), 1, 1, 1,
                             4, 3, -1,
                             3, 5, 3
   )
    //val yan = new MatrixD(r)
    val yan = new AirYan(r)
    yan.lud_npp_new(r)
    //yan.lud_ip()
    //val v = new VectorD(1,4,6)
    //yan.solveM(yan.lud_npp_new(r),v)
}