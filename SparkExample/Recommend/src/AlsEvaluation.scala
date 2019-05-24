import java.io.File
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ ALS, Rating, MatrixFactorizationModel }
import org.joda.time.format._
import org.joda.time._
import org.joda.time.Duration
import org.jfree.data.category.DefaultCategoryDataset
import org.apache.spark.mllib.regression.LabeledPoint

object AlsEvaluation {

  def main(args: Array[String]) {
    SetLogger
    println("==========資料準備階段===============")
    val (trainData, validationData, testData) = PrepareData()
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========訓練驗證階段===============")
    val bestModel = trainValidation(trainData, validationData)
    println("==========測試階段===============")
    val testRmse = computeRMSE(bestModel, testData)
    println("使用testData測試bestModel," + "結果rmse = " + testRmse)
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def trainValidation(trainData: RDD[Rating], validationData: RDD[Rating]): MatrixFactorizationModel = {
    println("-----評估 rank參數使用 ---------")
    evaluateParameter(trainData, validationData, "rank", Array(5, 10, 15, 20, 50, 100), Array(10), Array(0.1))
    println("-----評估 numIterations ---------")
    evaluateParameter(trainData, validationData, "numIterations", Array(10), Array(5, 10, 15, 20, 25), Array(0.1))
    println("-----評估 lambda ---------")
    evaluateParameter(trainData, validationData, "lambda", Array(10), Array(10), Array(0.05, 0.1, 1, 5, 10.0))
    println("-----所有參數交叉評估找出最好的參數組合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array(5, 10, 15, 20, 25), Array(5, 10, 15, 20, 25), Array(0.05, 0.1, 1, 5, 10.0))
    return (bestModel)
  }
  def evaluateParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                        evaluateParameter: String, rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]) =
    {

      var dataBarChart = new DefaultCategoryDataset()

      var dataLineChart = new DefaultCategoryDataset()
      for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) {

        val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)

        val parameterData =
          evaluateParameter match {
            case "rank"          => rank;
            case "numIterations" => numIterations;
            case "lambda"        => lambda
          }
        dataBarChart.addValue(rmse, evaluateParameter, parameterData.toString())
        dataLineChart.addValue(time, "Time", parameterData.toString())
      }

      Chart.plotBarLineChart("ALS evaluations " + evaluateParameter, evaluateParameter, "RMSE", 0.58, 5, "Time", dataBarChart, dataLineChart)
    }

  def evaluateAllParameter(trainData: RDD[Rating], validationData: RDD[Rating],
                           rankArray: Array[Int], numIterationsArray: Array[Int], lambdaArray: Array[Double]): MatrixFactorizationModel =
    {
      val evaluations =
        for (rank <- rankArray; numIterations <- numIterationsArray; lambda <- lambdaArray) yield {
          val (rmse, time) = trainModel(trainData, validationData, rank, numIterations, lambda)
          (rank, numIterations, lambda, rmse)
        }
      val Eval = (evaluations.sortBy(_._4))
      val BestEval = Eval(0)
      println("最佳model參數：rank:" + BestEval._1 + ",iterations:" + BestEval._2 + "lambda" + BestEval._3 + ",結果rmse = " + BestEval._4)
      val bestModel = ALS.train(trainData, BestEval._1, BestEval._2, BestEval._3)
      (bestModel)
    }
  def PrepareData(): (RDD[Rating], RDD[Rating], RDD[Rating]) = {

    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    //----------------------1.建立使用者評價資料-------------
    print("開始讀取使用者評價資料中...")
    val DataDir = "data"
    val rawUserData = sc.textFile(new File(DataDir, "u.data").toString)

    val rawRatings = rawUserData.map(_.split("\t").take(3))

    val ratingsRDD = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    println("共計：" + ratingsRDD.count.toString() + "筆ratings")

    //----------------------2.建立電影ID與名稱對照表-------------
    print("開始讀取電影資料中...")
    val itemRDD = sc.textFile(new File(DataDir, "u.item").toString)
    val movieTitle = itemRDD.map(line => line.split("\\|").take(2))
      .map(array => (array(0).toInt, array(1))).collect().toMap
    //----------------------3.列印資料-------------
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(_.user).distinct().count()
    val numMovies = ratingsRDD.map(_.product).distinct().count()
    println("共計：ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)
    //----------------------3.以隨機方式將資料分為3部份並且回傳-------------

    println("將資料分為")
    val Array(trainData, validationData, testData) = ratingsRDD.randomSplit(Array(0.8, 0.1, 0.1))

    println("  trainData:" + trainData.count() + "  validationData:" + validationData.count() + "  testData:" + testData.count())
    return (trainData, validationData, testData)
  }

  def trainModel(trainData: RDD[Rating], validationData: RDD[Rating], rank: Int, iterations: Int, lambda: Double): (Double, Double) = {
    val startTime = new DateTime()
    val model = ALS.train(trainData, rank, iterations, lambda)
    val endTime = new DateTime()
    val Rmse = computeRMSE(model, validationData)
    val duration = new Duration(startTime, endTime)
    println(f"訓練參數：rank:$rank%3d,iterations:$iterations%.2f ,lambda = $lambda%.2f 結果 Rmse=$Rmse%.2f" + "訓練需要時間" + duration.getMillis + "毫秒")
    (Rmse, duration.getStandardSeconds)
  }

  def computeRMSE(model: MatrixFactorizationModel, RatingRDD: RDD[Rating]): Double = {

    val num = RatingRDD.count()
    val predictedRDD = model.predict(RatingRDD.map(r => (r.user, r.product)))
    val predictedAndRatings =
      predictedRDD.map(p => ((p.user, p.product), p.rating))
        .join(RatingRDD.map(r => ((r.user, r.product), r.rating)))
        .values
    math.sqrt(predictedAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / num)
  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

}
