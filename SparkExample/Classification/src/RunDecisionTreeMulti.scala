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

object RunDecisionTreeMulti {

  def main(args: Array[String]): Unit = {
    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("RunDecisionTreeMulti")
    println("==========資料準備階段===============")
    val (trainData, validationData, testData) = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========訓練評估階段===============")

    println()
    print("是否需要進行參數調校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData)
      println("==========測試階段===============")
      val precise = evaluateModel(model, testData)
      println("使用testata測試,結果 precise:" + precise)
      println("==========預測資料===============")
      PredictData(sc, model)
    } else {
      val model = trainEvaluate(trainData, validationData)
      println("==========測試階段===============")
      val precise = evaluateModel(model, testData)
      println("使用testata測試,結果 precise:" + precise)
      println("==========預測資料===============")
      PredictData(sc, model)
    }
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //----------------------1.匯入轉換資料-------------
    print("開始匯入資料...")
    val rawData = sc.textFile("data/covtype.data") 
    println("共計：" + rawData.count.toString() + "筆") 
    //----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    println("準備訓練資料...")
    val labelpointRDD = rawData.map { record =>
      val fields = record.split(',').map(_.toDouble)
      val label = fields.last - 1
      LabeledPoint(label, Vectors.dense(fields.init))
    }
    //----------------------3.以隨機方式將資料分為3部份並且回傳-------------
    val Array(trainData, validationData, testData) = labelpointRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("將資料分為 trainData:" + trainData.count() + "   cvData:" + validationData.count() + "   testData:" + testData.count())
    return (trainData, validationData, testData) 
  }

  def PredictData(sc: SparkContext, model: DecisionTreeModel): Unit = {
    val rawData = sc.textFile("data/covtype.data") 
    println("共計：" + rawData.count.toString() + "筆") 
    println("準備測試資料...")
    val Array(pData, oData) = rawData.randomSplit(Array(0.1, 0.9))
    val data = pData.take(20).map { record =>
      val fields = record.split(',').map(_.toDouble)
      val features = Vectors.dense(fields.init)
      val label = fields.last - 1
      val predict = model.predict(features)
      val result = (if (label == predict) "正確" else "錯誤")
      println("土地條件：海拔:" + features(0) + " 方位:" + features(1) + " 斜率:" + features(2) + " 水源垂直距離:" + features(3) + " 水源水平距離:" + features(4) + " 9點時陰影:" + features(5) + "....==>預測:" + predict + " 實際:" + label + "結果:" + result)
    }
  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    print("開始訓練...")
    val (model, time) = trainModel(trainData, "entropy", 20, 100)
    println("訓練完成,所需時間:" + time + "毫秒")
    val precision = evaluateModel(model, validationData)
    println("評估結果precision=" + precision)
    return (model)
  }

  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }

  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }

    val Metrics = new MulticlassMetrics(scoreAndLabels)
    val precision = Metrics.precision
    (precision)
  }
  def testModel(model: DecisionTreeModel, testData: RDD[LabeledPoint]): Unit = {
    val precise = evaluateModel(model, testData)
    println("使用testata測試,結果 precise:" + precise)
    println("最佳模型使用testData前50筆資料進行預測:")
    val PredictData = testData.take(50) 
    PredictData.foreach { data =>     
      val predict = model.predict(data.features) 
      val result = (if (data.label == predict) "正確" else "錯誤") 
      println("實際結果:" + data.label + "預測結果:" + predict + result + data.features)
    }

  }

  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    println("-----評估 Impurity參數使用 gini, entropy---------")
    evaluateParameter(trainData, validationData, "impurity", Array("gini", "entropy"), Array(10), Array(10))
    println("-----評估MaxDepth參數使用 (3, 5, 10, 15, 20)---------")
    evaluateParameter(trainData, validationData, "maxDepth", Array("gini"), Array(3, 5, 10, 15, 20, 25), Array(10))
    println("-----評估maxBins參數使用 (3, 5, 10, 50, 100)---------")
    evaluateParameter(trainData, validationData, "maxBins", Array("gini"), Array(10), Array(3, 5, 10, 50, 100, 200))
    println("-----所有參數交叉評估找出最好的參數組合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array("gini", "entropy"),
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
        val precise = evaluateModel(model, validationData)
        val parameterData = 
          evaluateParameter match {
            case "impurity" => impurity;
            case "maxDepth" => maxDepth;
            case "maxBins"  => maxBins
          }
        dataBarChart.addValue(precise, evaluateParameter, parameterData.toString())
        dataLineChart.addValue(time, "Time", parameterData.toString())
      }
      Chart.plotBarLineChart("DecisionTree evaluations " + evaluateParameter, evaluateParameter, "precision", 0.6, 1, "Time", dataBarChart, dataLineChart)
    }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel =
    {
      val evaluationsArray =
        for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) yield {
          val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins) 
          val precise = evaluateModel(model, validationData) 
          (impurity, maxDepth, maxBins, precise)
        }
      val BestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
      println("調校後最佳參數：impurity:" + BestEval._1 + "  ,maxDepth:" + BestEval._2 + "  ,maxBins:" + BestEval._3
        + "  ,結果precise = " + BestEval._4)
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