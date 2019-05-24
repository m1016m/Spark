import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd.RDD


object RunWordCount {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")

    println("開始執行RunWordCount")
  
    val sc = new SparkContext(new SparkConf().setAppName("wordCount").setMaster("local[4]"))

    println("開始讀取文字檔...")
    val textFile = sc.textFile("data/pg5000.txt") 

    println("開始建立RDD...")
    val countsRDD = textFile.flatMap(line => line.split(" ")) 
      .map(word => (word, 1))
      .reduceByKey(_ + _) 

    println("開始儲存至文字檔...")
    try {
      countsRDD.saveAsTextFile("data/output") 
      println("已經存檔成功")
    } catch {
      case e: Exception => println("輸出目錄已經存在,請先刪除原有目錄");
    }

  }
}