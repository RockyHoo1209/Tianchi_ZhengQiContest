import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.FloatType
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.feature.MinMaxScalerModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{
  RandomForestRegressionModel,
  RandomForestRegressor
}
import com.microsoft.ml.spark.lightgbm.{
  LightGBMRegressor,
  LightGBMRegressionModel
}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

// utf for (object's)function need to Serializable
object ScalaApp extends Serializable {
  spark.conf.set("broadcastTimeout", -1)
//   at first you should upload the dataset on hdfs
  val train_path = "hdfs://master:9000/zhengqi_train.txt"
  val test_path = "hdfs://master:9000/zhengqi_test.txt"
  val train_item = spark.read
    .option("header", true)
    .option("sep", "\t")
    .option("inferSchema", true)
    .option("encoding", "UTF-8")
    .csv(train_path)

  var target = train_item.select("target")
  var train_features = train_item.drop("target")

  val test_item = spark.read
    .option("header", true)
    .option("sep", "\t")
    .option("inferSchema", true)
    .option("encoding", "UTF-8")
    .csv(test_path)

  var all_features = test_item.union(train_features)

  def MinMax(data: DataFrame): MinMaxScalerModel = {
    val assembler =
      new VectorAssembler().setInputCols(data.columns).setOutputCol("features")
    val scaler =
      new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    return scaler
      .fit(assembler.transform(data))
  }

  def SelectFeatures(
      feature_col: String,
      label_col: String,
      top_features: Int,
      train_data: DataFrame
  ): Array[Int] = {
    val selector = new ChiSqSelector()
      .setNumTopFeatures(top_features)
      .setFeaturesCol(feature_col)
      .setLabelCol(label_col)
      .setOutputCol("selectedFeatures")
    val select_model = selector.fit(train_data)
    return select_model.selectedFeatures
  }

  // Generate index col for later join operation
  var index_counter = 0
  def GenIndex(x: Double): Int = {
    index_counter += 1
    return index_counter
  }

  def MergeDF(first_df: DataFrame, second_df: DataFrame): DataFrame = {
    val addCol = udf(GenIndex _)
    var first = first_df
    var second = second_df
    first = first.withColumn("zeros", lit(0.0))
    second = second.withColumn("zeros", lit(0.0))
    first = first.withColumn("index", addCol(first("zeros")))
    second = second.withColumn("index", addCol(second("zeros")))
    val join_express = first.col("index") === second.col("index")
    first = first.join(second, join_express)
    return first.drop("index").drop("zeros")
  }

  //ensure installed mmlspark
  def LGB(
      train_df: DataFrame,
      input_col: String,
      label_col: String,
      predict_col: String
  ): CrossValidatorModel = {
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > max_categories values are treated as continuous
    var train_data = train_df
    val featureIndexer = new VectorIndexer()
      .setInputCol(input_col)
      .setOutputCol("indexedFeatures")
      .fit(train_data)
    // Train a RandomForest model.
    val lgb = new LightGBMRegressor()
      .setLabelCol(label_col)
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, lgb))

    // Train model. This also runs the indexer.
    // val model = pipeline.fit(train_data)
    val model = K_fold(train_data, 7, label_col, predict_col, "rmse", pipeline)
    return model
  }

  def RF_Model(
      train_df: DataFrame,
      input_col: String,
      max_categories: Int,
      label_col: String,
      num_trees: Int,
      predict_col: String
  ): CrossValidatorModel = {
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > max_categories values are treated as continuous
    var train_data = train_df
    val featureIndexer = new VectorIndexer()
      .setInputCol(input_col)
      .setOutputCol("indexedFeatures")
      .setMaxCategories(max_categories)
      .fit(train_data)
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol(label_col)
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(num_trees)

    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rf))

    // Train model. This also runs the indexer.
    // val model = pipeline.fit(train_data)
    val model = K_fold(train_data, 7, label_col, predict_col, "rmse", pipeline)
    return model
  }

  def K_fold(
      train_data: DataFrame,
      num_folds: Int,
      label: String,
      predictionCol: String,
      metricName: String,
      pipeline: Pipeline
  ): CrossValidatorModel = {

    val params = new ParamGridBuilder().build()
    val evaluator = new RegressionEvaluator()
      .setMetricName(metricName)
      .setPredictionCol(predictionCol)
      .setLabelCol(label)
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(params)
      .setNumFolds(num_folds)
    val model = cv.fit(train_data)
    return model
  }

  // only 2 model
  def PredictionAverage(pred1: DataFrame, pred2: DataFrame): DataFrame = {
    val pred1_list = pred1.collect.map(x => x.get(0)).toList
    val pred2_list = pred2.collect.map(x => x.get(0)).toList
    // since origin is [Any] type ,we should transform to the [Double] Tpe
    val pred_sum =
      (pred1_list, pred2_list).zipped.map(_.toString.toDouble + _.toString.toDouble)
    val pred_average = pred_sum.map(x => x / 2)
    return pred_average.toDF
  }

  val train_assembler2 =
    new VectorAssembler()
      .setInputCols(train_item.columns.slice(0, test_item.columns.length - 1))
      .setOutputCol("features")

  var features =
    SelectFeatures(
      "features",
      "target",
      30,
      train_assembler2.transform(train_item)
    ).map(x => 'V' + String.valueOf(x))
  train_features = train_features.select(features.head, features.tail: _*)
  var test_features = test_item.select(features.head, features.tail: _*)
  all_features = all_features.select(features.head, features.tail: _*)

  val minmax_model = MinMax(all_features)

  val train_assembler =
    new VectorAssembler()
      .setInputCols(train_features.columns)
      .setOutputCol("features")

  val test_assembler =
    new VectorAssembler()
      .setInputCols(test_features.columns)
      .setOutputCol("features")

  train_features = minmax_model
    .transform(train_assembler.transform(train_features))
    .select("scaledFeatures")
  test_features = minmax_model
    .transform(test_assembler.transform(test_features))
    .select("scaledFeatures")

  // split the train_data into offline train and validate
  var train_features_target = MergeDF(train_features, target)
  val splits = train_features_target.randomSplit(Array(0.7, 0.3))
  var (trainingData, validData) = (splits(0), splits(1))
  val rf_model =
    RF_Model(trainingData, "scaledFeatures", 32, "target", 500, "target")
  val lgb_model = LGB(trainingData, "scaledFeatures", "target", "target")

  val rf_pred = rf_model.transform(test_features).select("prediction")
  val lgb_pred = lgb_model.transform(test_features).select("prediction")
  val average_pred = PredictionAverage(rf_pred, lgb_pred)

  //transform double to string
  var pred_ans =
    average_pred.select(
      average_pred.columns.map(c => col(c).cast(StringType)): _*
    )
  // save as one .txt file
  pred_ans.rdd
    .map(x => x.mkString(","))
    .repartition(1)
    .saveAsTextFile("file:/usr/work/Spark_Code/Zhengqi_predict/ans_pred")

}
