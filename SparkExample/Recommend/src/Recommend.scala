import java.io.File
import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ ALS, Rating, MatrixFactorizationModel }
 object Recommend {
  def main(args: Array[String]) {
    SetLogger
    println("==========資料準備階段===============")
    val (ratings, movieTitle) = PrepareData()
    println("==========訓練階段===============")
    print("開始使用 " + ratings.count() + "筆評比資料進行訓練模型... ")
    val model = ALS.train(ratings, 5, 20, 0.1) 
    println("訓練完成!")
    println("==========推薦階段===============")
    recommend(model, movieTitle)
    println("完成")
  }

    def recommend(model: MatrixFactorizationModel, movieTitle: Map[Int, String]) = {
    var choose = ""
    while (choose != "3") { //如果選擇3.離開,就結束執行程式
      print("請選擇要推薦類型  1.針對使用者推薦電影 2.針對電影推薦有興趣的使用者 3.離開?")
      choose = readLine() //讀取使用者輸入
      if (choose == "1") { //如果輸入1.針對使用者推薦電影
        print("請輸入使用者id?")
        val inputUserID = readLine() //讀取使用者ID
        RecommendMovies(model, movieTitle, inputUserID.toInt) //針對此使用者推薦電影
      } else if (choose == "2") { //如果輸入2.針對電影推薦有興趣的使用者
        print("請輸入電影的 id?")
        val inputMovieID = readLine() //讀取MovieID
        RecommendUsers(model, movieTitle, inputMovieID.toInt) //針對此電影推薦使用者
      }
    }
  }


  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }

  def PrepareData(): (RDD[Rating], Map[Int, String]) = {

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
    val numRatings = ratingsRDD.count() 
    val numUsers = ratingsRDD.map(_.user).distinct().count() 
    val numMovies = ratingsRDD.map(_.product).distinct().count() 
    println("共計：ratings: " + numRatings + " User " + numUsers + " Movie " + numMovies)
    return (ratingsRDD, movieTitle)
  }

  def RecommendMovies(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputUserID: Int) = {
    val RecommendMovie = model.recommendProducts(inputUserID, 10) 
    var i = 1
    println("針對使用者id" + inputUserID + "推薦下列電影:")
    RecommendMovie.foreach { r => 
      println(i.toString() + "." + movieTitle(r.product) + "評價:" + r.rating.toString())
      i += 1
    }
  }

  def RecommendUsers(model: MatrixFactorizationModel, movieTitle: Map[Int, String], inputMovieID: Int) = {
    val RecommendUser = model.recommendUsers(inputMovieID, 10) 
    var i = 1
    println("針對電影 id" + inputMovieID + "電影名:" + movieTitle(inputMovieID.toInt) + "推薦下列使用者id:")
    RecommendUser.foreach { r => 
      println(i.toString + "使用者id:" + r.user + "   評價:" + r.rating)
      i = i + 1
    }
  }

}
