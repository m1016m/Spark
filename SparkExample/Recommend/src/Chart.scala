import org.jfree.chart._
import org.jfree.data.xy._
import org.jfree.data.category.DefaultCategoryDataset
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.axis._
import java.awt.Color
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.labels.StandardCategoryToolTipGenerator;
import java.awt.BasicStroke

object Chart {
  def plotBarLineChart(Title: String, xLabel: String, yBarLabel: String, yBarMin: Double, yBarMax: Double, yLineLabel: String, dataBarChart : DefaultCategoryDataset, dataLineChart: DefaultCategoryDataset): Unit = {
    //畫出Bar Chart
    val chart = ChartFactory
         .createBarChart(  
        "", // Bar Chart 標題
        xLabel, // X軸標題
        yBarLabel, // Bar Chart 標題 y軸標題l
        dataBarChart , // Bar Chart資料
        org.jfree.chart.plot.PlotOrientation.VERTICAL,//畫圖方向垂直 
        true, // 包含 legend
        true, // 顯示tooltips 
        false // 不要URL generator
        );
    //取得plot  
    val plot = chart.getCategoryPlot();
    plot.setBackgroundPaint(new Color(0xEE, 0xEE, 0xFF));
    plot.setDomainAxisLocation(AxisLocation.BOTTOM_OR_RIGHT);
    plot.setDataset(1, dataLineChart); plot.mapDatasetToRangeAxis(1, 1)
    //畫長條圖y軸
    val vn = plot.getRangeAxis(); vn.setRange(yBarMin, yBarMax);  vn.setAutoTickUnitSelection(true)
    //畫折線圖y軸       
    val axis2 = new NumberAxis(yLineLabel); plot.setRangeAxis(1, axis2);
    val renderer2 = new LineAndShapeRenderer()
    renderer2.setToolTipGenerator(new StandardCategoryToolTipGenerator());
    //設定先畫長條圖,再畫折線圖以免折線圖被蓋掉 
    plot.setRenderer(1, renderer2);plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);
    //建立畫框    
    val frame = new ChartFrame(Title,chart); frame.setSize(500, 500);
    frame.pack(); frame.setVisible(true)

  }

}