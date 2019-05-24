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

object RunDecisionTreeBinary {

  def main(args: Array[String]): Unit = {
    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[4]"))
    println("RunDecisionTreeBinary")
    println("==========資料準備階段===============")
    val (trainData, validationData, testData, categoriesMap) = PrepareData(sc)
    trainData.persist(); validationData.persist(); testData.persist()
    println("==========訓練評估階段===============")
    println()
    print("是否需要進行參數調校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData)
      println("==========測試階段===============")
      val auc = evaluateModel(model, testData)
      println("使用testata測試最佳模型,結果 AUC:" + auc)
      println("==========預測資料===============")
      PredictData(sc, model, categoriesMap)
    } else {
      val model = trainEvaluate(trainData, validationData)
      println("==========測試階段===============")
      val auc = evaluateModel(model, testData)
      println("使用testata測試最佳模型,結果 AUC:" + auc)
      println("==========預測資料===============")
      PredictData(sc, model, categoriesMap)
    }

    //取消暫存在記憶體中
    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], Map[String, Int]) = {
    //----------------------1.匯入並轉換資料-------------
    print("開始匯入資料...")
    val rawDataWithHeader = sc.textFile("data/train.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val lines = rawData.map(_.split("\t"))
    println("共計：" + lines.count.toString() + "筆")
    //----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    val categoriesMap = lines.map(fields => fields(3)).distinct.collect.zipWithIndex.toMap
    val labelpointRDD = lines.map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      val label = trFields(fields.size - 1).toInt
      LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
    }
    //----------------------3.以隨機方式將資料分為3部份並且回傳-------------

    val Array(trainData, validationData, testData) = labelpointRDD.randomSplit(Array(8, 1, 1))

    println("將資料分trainData:" + trainData.count() + "   validationData:" + validationData.count()
      + "   testData:" + testData.count())

    return (trainData, validationData, testData, categoriesMap) //回傳資料
  }

  def PredictData(sc: SparkContext, model: DecisionTreeModel, categoriesMap: Map[String, Int]): Unit = {
    //----------------------1.匯入並轉換資料-------------
    val rawDataWithHeader = sc.textFile("data/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val lines = rawData.map(_.split("\t"))
    println("共計：" + lines.count.toString() + "筆")
    //----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    val dataRDD = lines.take(20).map { fields =>
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trFields.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      val label = 0
      //----------------------3進行預測-------------
      val url = trFields(0)
      val Features = Vectors.dense(categoryFeaturesArray ++ numericalFeatures)
      val predict = model.predict(Features).toInt
      var predictDesc = { predict match { case 0 => "暫時性網頁(ephemeral)"; case 1 => "長青網頁(evergreen)"; } }
      println(" 網址：  " + url + "==>預測:" + predictDesc)
    }

  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    print("開始訓練...")
    val (model, time) = trainModel(trainData, "entropy", 5, 5)
    println("訓練完成,所需時間:" + time + "毫秒")
    val AUC = evaluateModel(model, validationData)
    println("評估結果AUC=" + AUC)
    return (model)
  }
  //訓練模型
  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    val model = DecisionTree.trainClassifier(trainData, 2, Map[Int, Int](), impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }
  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val AUC = Metrics.areaUnderROC
    //傳回AUC
    (AUC)
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
        val auc = evaluateModel(model, validationData)

        val parameterData =
          evaluateParameter match {
            case "impurity" => impurity;
            case "maxDepth" => maxDepth;
            case "maxBins"  => maxBins
          }
        dataBarChart.addValue(auc, evaluateParameter, parameterData.toString())
        dataLineChart.addValue(time, "Time", parameterData.toString())
      }
      Chart.plotBarLineChart("DecisionTree evaluations " + evaluateParameter, evaluateParameter, "AUC", 0.58, 0.7, "Time", dataBarChart, dataLineChart)
    }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel =
    {
      val evaluationsArray =
        for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) yield {
          val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
          val auc = evaluateModel(model, validationData)
          (impurity, maxDepth, maxBins, auc)
        }
      val BestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
      println("調校後最佳參數：impurity:" + BestEval._1 + "  ,maxDepth:" + BestEval._2 + "  ,maxBins:" + BestEval._3
        + "  ,結果AUC = " + BestEval._4)
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