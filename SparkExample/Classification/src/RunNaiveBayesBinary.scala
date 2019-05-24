import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler

import org.joda.time._
import org.jfree.data.category.DefaultCategoryDataset

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel

object RunNaiveBayesBinary {

  def main(args: Array[String]): Unit = {

    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("RunNaiveBayesBinary")
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

    trainData.unpersist(); validationData.unpersist(); testData.unpersist()
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], Map[String, Int]) = {
    //----------------------1.匯入轉換資料-------------
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
        .map(d => if (d < 0) 0.0 else d) 
      val label = trFields(fields.size - 1).toInt
      LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
    }

    val featuresData = labelpointRDD.map(labelpoint => labelpoint.features)
    val stdScaler = new StandardScaler(withMean = false, withStd = true).fit(featuresData)
    val scaledRDD = labelpointRDD.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)))
    //----------------------3.以隨機方式將資料分為3部份並且回傳-------------
    val Array(trainData, validationData, testData) = scaledRDD.randomSplit(Array(0.8, 0.1, 0.1))
    println("將資料分trainData:" + trainData.count() + "   validationData:" + validationData.count() + "   testData:" + testData.count())
    return (trainData, validationData, testData, categoriesMap) //回傳資料
  }

  def PredictData(sc: SparkContext, model: NaiveBayesModel, categoriesMap: Map[String, Int]): Unit = {

    //----------------------1.匯入轉換資料-------------
    print("開始匯入資料...")
    val rawDataWithHeader = sc.textFile("data/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val lines = rawData.map(_.split("\t"))
    println("共計：" + lines.count.toString() + "筆")
    //----------------------2.建立訓練評估所需資料 RDD[LabeledPoint]-------------
    val labelpointRDD = lines.map { fields =>
      val trimmed = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      categoryFeaturesArray(categoryIdx) = 1
      val numericalFeatures = trimmed.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)
      val label = 0
      val url = trimmed(0)
      (LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures)), url)
    }
    val featuresRDD = labelpointRDD.map { case (labelpoint, url) => labelpoint.features }
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresRDD)
    val scaledRDD = labelpointRDD.map { case (labelpoint, url) => (LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)), url) }
    scaledRDD.take(10).map {
      case (labelpoint, url) =>
        val predict = model.predict(labelpoint.features)
        var predictDesc = { predict match { case 0 => "暫時性網頁(ephemeral)"; case 1 => "長青網頁(evergreen)"; } }
        println(" 網址：  " + url + "==>預測:" + predictDesc)
    }

  }

  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): NaiveBayesModel = {
    print("開始訓練...")
    val (model, time) = trainModel(trainData, 5)
    println("訓練完成,所需時間:" + time + "毫秒")
    val AUC = evaluateModel(model, validationData)
    println("評估結果AUC=" + AUC)
    return (model)
  }

  def trainModel(trainData: RDD[LabeledPoint], lambda: Int): (NaiveBayesModel, Double) = {
    val startTime = new DateTime()
    val model = NaiveBayes.train(trainData, lambda)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis()) 
  }
  def evaluateModel(model: NaiveBayesModel, validationData: RDD[LabeledPoint]): (Double) = {
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels) 
    val AUC = Metrics.areaUnderROC 
    (AUC)
  }
  def predict(model: NaiveBayesModel, testData: RDD[LabeledPoint]): Unit = {
    println("最佳模型使用testData前10筆資料進行預測:")
    val testPredictData = testData.take(10)
    testPredictData.foreach { test =>
      val predict = model.predict(test.features)
      val result = (if (test.label == predict) "正確" else "錯誤")
      println("實際結果:" + test.label + "預測結果:" + predict + result + test.features)
    }
  }

  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): NaiveBayesModel = {
    println("-----評估 lambda參數使用 1, 3, 5, 15, 25---------")
    evaluateParameter(trainData, validationData, "lambda", Array(1, 3, 5, 15, 25))

    println("-----所有參數交叉評估找出最好的參數組合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array(1, 3, 5, 15, 25))
    return (bestModel)
  }
  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint],
                        evaluateParameter: String, lambdaArray: Array[Int]) =
    {
      var dataBarChart = new DefaultCategoryDataset()
      var dataLineChart = new DefaultCategoryDataset()
      for (lambda <- lambdaArray) {
        val (model, time) = trainModel(trainData, lambda)
        val auc = evaluateModel(model, validationData)
        val parameterData =
          evaluateParameter match {
            case "lambda" => lambda;
          }
        dataBarChart.addValue(auc, evaluateParameter, parameterData.toString())
        dataLineChart.addValue(time, "Time", parameterData.toString())
      }
      Chart.plotBarLineChart("NaiveBayes evaluations " + evaluateParameter, evaluateParameter, "AUC", 0.48, 0.7, "Time", dataBarChart, dataLineChart)
    }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], lambdaArray: Array[Int]): NaiveBayesModel =
    {
      val evaluations =
        for (lambda <- lambdaArray) yield {
          val (model, time) = trainModel(trainData, lambda)
          val auc = evaluateModel(model, validationData)
          (lambda, auc)
        }
      val BestEval = (evaluations.sortBy(_._2).reverse)(0)
      println("調校後最佳參數：lambda:" + BestEval._1 + "  ,結果AUC = " + BestEval._2)

      val bestModel = NaiveBayes.train(
        trainData.union(validationData), BestEval._1)
      return bestModel
    }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}