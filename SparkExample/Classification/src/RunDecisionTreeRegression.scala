
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.joda.time._
import org.jfree.data.category.DefaultCategoryDataset

object RunDecisionTreeRegression {

  def main(args: Array[String]): Unit = {

    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("RunDecisionTreeRegression")
    println("==========資料準備階段===============")

    val (trainData, validationData, testData) = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========訓練評估階段===============")
    println()
    print("是否需要進行參數調校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData)
      println("==========測試階段===============")
      val RMSE = evaluateModel(model, testData)
      println("使用testata測試,結果 RMSE:" + RMSE)
      println("==========預測資料===============")
      PredictData(sc, model)
    } else {
      val model = trainEvaluate(trainData, validationData)
      println("==========測試階段===============")
      val RMSE = evaluateModel(model, testData)
      println("使用testata測試,結果 RMSE:" + RMSE)
      println("==========預測資料===============")
      PredictData(sc, model)
    }

    //取消暫存在記憶體中
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //----------------------1.匯入轉換資料-------------
    print("開始匯入資料...")
    val rawDataWithHeader = sc.textFile("data/hour.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    println("共計：" + rawData.count.toString() + "筆")
    //----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    println("準備訓練資料...")
    val records = rawData.map(line => line.split(","))
    val data = records.map { fields =>
      val label = fields(fields.size - 1).toInt
      val featureSeason = fields.slice(2, 3).map(d => d.toDouble)
      val features = fields.slice(4, fields.size - 3).map(d => d.toDouble)
      LabeledPoint(label, Vectors.dense(featureSeason ++ features))
    }
    //----------------------3.以隨機方式將資料分為3部份並且回傳------------
    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    println("將資料分為 trainData:" + trainData.count() + "   cvData:" + cvData.count() + "   testData:" + testData.count())
    trainData.persist(); cvData.persist(); testData.persist()
    return (trainData, cvData, testData)
  }

  def PredictData(sc: SparkContext, model: DecisionTreeModel): Unit = {
    //----------------------1.匯入轉換資料-------------

    print("開始匯入資料...")
    val rawDataWithHeader = sc.textFile("data/hour.csv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    println("共計：" + rawData.count.toString() + "筆") 
    //----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    println("準備訓練資料...")
    val Array(pData, oData) = rawData.randomSplit(Array(0.1, 0.9))
    val records = pData.map(line => line.split(","))
    val data = records.take(200).map { fields =>

      val label = fields(fields.size - 1).toInt
      val featureSeason = fields.slice(2, 3).map(d => d.toDouble)
      val features = fields.slice(4, fields.size - 3).map(d => d.toDouble)

      val featuresVectors = Vectors.dense(featureSeason ++ features)

      var dataDesc = ""
      dataDesc = dataDesc + { featuresVectors(0) match { case 1 => "春"; case 2 => "夏"; case 3 => "秋"; case 4 => "冬"; } } + "天,"
      dataDesc = dataDesc + featuresVectors(1).toInt + "月,"
      dataDesc = dataDesc + featuresVectors(2).toInt + "時,"
      dataDesc = dataDesc + { featuresVectors(3) match { case 0 => "非假日"; case 1 => "假日"; } } + ","
      dataDesc = dataDesc + "星期" + { featuresVectors(4) match { case 0 => "日"; case 1 => "一"; case 2 => "二"; case 3 => "三"; case 4 => "四"; case 5 => "五"; case 6 => "六"; } } + ","
      dataDesc = dataDesc + { featuresVectors(5) match { case 1 => "工作日"; case 0 => "非工作日" } } + ","
      dataDesc = dataDesc + { featuresVectors(6) match { case 1 => "晴"; case 2 => "陰"; case 3 => "小雨"; case 4 => "大雨" } } + ","
      dataDesc = dataDesc + (featuresVectors(7) * 41).toInt + "度,"
      dataDesc = dataDesc + "體感" + (featuresVectors(8) * 50).toInt + "度,"
      dataDesc = dataDesc + "溼度" + (featuresVectors(9) * 100).toInt + ","
      dataDesc = dataDesc + "風速" + (featuresVectors(10) * 67).toInt + ","

      val predict = model.predict(featuresVectors) 
      val result = (if (label == predict) "正確" else "錯誤") 
      val error = (math.abs(label - predict).toString())
      println("  特徵: " + dataDesc + " ==> 預測結果:" + predict.toInt + "    實際:" + label.toInt + "  誤差:" + error)
    }

  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    print("開始訓練...")
    val (model, time) = trainModel(trainData, "variance", 10, 50)
    println("訓練完成,所需時間:" + time + "毫秒")
    val RMSE = evaluateModel(model, validationData)
    println("評估結果RMSE=" + RMSE)
    return (model)
  }
  
  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainRegressor(trainData, Map[Int, Int](), impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }
  
  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }
    val Metrics = new RegressionMetrics(scoreAndLabels)
    val RMSE = Metrics.rootMeanSquaredError
    (RMSE)
  }

  def testModel(model: DecisionTreeModel, testData: RDD[LabeledPoint]): Unit = {
    val RMSE = evaluateModel(model, testData)
    println("使用testata測試,結果 RMSE:" + RMSE)
    println("最佳模型使用testData前50筆資料進行預測:")
    val PredictData = testData.take(50)  
    PredictData.foreach { data =>     
      val predict = model.predict(data.features) 
      val result = (if (data.label == predict) "正確" else "錯誤") 
      val error = math.abs(data.label - predict).toString()
      println("  資料: " + data.features + "  實際:" + data.label + "  預測:" + predict + "誤差:" + error)
    }
 }

  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {

    println("-----評估MaxDepth參數使用 (3, 5, 10, 15, 20)---------")
    evaluateParameter(trainData, validationData, "maxDepth", Array("variance"), Array(3, 5, 10, 15, 20, 25), Array(10))
    println("-----評估maxBins參數使用 (3, 5, 10, 50, 100)---------")
    evaluateParameter(trainData, validationData, "maxBins", Array("variance"), Array(10), Array(3, 5, 10, 50, 100, 200))
    println("-----所有參數交叉評估找出最好的參數組合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array("variance"),
      Array(3, 5, 10, 15, 20), Array(3, 5, 10, 50, 100))
    return (bestModel)
  }
  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint],
                        evaluateParameter: String, impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]) =
    {
      var dataBarChart = new DefaultCategoryDataset()
      var dataLineChart = new DefaultCategoryDataset()
      for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) {
        val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins) 
        val RMSE = evaluateModel(model, validationData) 
        val parameterData = 
          evaluateParameter match {
            case "impurity" => impurity;
            case "maxDepth" => maxDepth;
            case "maxBins"  => maxBins
          }
        dataBarChart.addValue(RMSE, evaluateParameter, parameterData.toString())
        dataLineChart.addValue(time, "Time", parameterData.toString())
      }
      Chart.plotBarLineChart("DecisionTree evaluations " + evaluateParameter, evaluateParameter, "RMSE", 0, 150, "Time", dataBarChart, dataLineChart)
    }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel =
    {
      val evaluationsArray =
        for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) yield {
          val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins) 
          val RMSE = evaluateModel(model, validationData) 
      
          (impurity, maxDepth, maxBins, RMSE)
        }
      val evaluationsArraySortedAsc = (evaluationsArray.sortBy(_._4))
      val BestEval = evaluationsArraySortedAsc(0)
      println("調校後最佳參數：impurity:" + BestEval._1 + "  ,maxDepth:" + BestEval._2 + "  ,maxBins:" + BestEval._3
        + "  ,結果RMSE = " + BestEval._4)
      val (bestModel, time) = trainModel(trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3)
      return bestModel
    }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}