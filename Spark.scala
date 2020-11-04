

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
val titanic_data =spark.read.format("csv")
.option("header", "true")
.option("inferSchema", "true")
.load("/FileStore/tables/train.csv")
titanic_data.printSchema
val titanic_data1 = titanic_data.select('Survived.as("label"), 'Pclass.as("ticket_class"), 'Sex.as("gender"),'Age.as("age")).filter('age.isNotNull)

val Array(training, test) = titanic_data1.randomSplit(Array(0.8,0.2))
println(s"training count: ${training.count}, test count: ${test.count}")

val genderIndxr = new StringIndexer().setInputCol("gender").setOutputCol("genderIdx")
val assembler = new VectorAssembler().setInputCols(Array("ticket_class", "genderIdx", "age")).setOutputCol("features")

val logisticRegression = new LogisticRegression().setFamily("binomial")

val pipeline = new Pipeline().setStages(Array(genderIndxr,assembler,logisticRegression))
val model = pipeline.fit(training)

val predictions = model.transform(test)
val evaluator = new BinaryClassificationEvaluator()
evaluator.evaluate(predictions)
evaluator.getMetricName



import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.RegressionMetrics

val house_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/train-1.csv")
val cols = Seq[String]("SalePrice", "LotArea","RoofStyle", "Heating", "1stFlrSF","2ndFlrSF","BedroomAbvGr", "KitchenAbvGr", "GarageCars", "TotRmsAbvGrd","YearBuilt")
val colNames = cols.map(n => col(n))
val skinny_house_data = house_data.select(colNames:_*)
val skinny_house_data1 = skinny_house_data.withColumn("TotalSF", col("1stFlrSF") +
                                                           col("2ndFlrSF"))
                                                          .drop("1stFlrSF", "2ndFlrSF")
                     .withColumn("SalePrice", $"SalePrice".cast("double"))
skinny_house_data1.describe("SalePrice").show

val roofStyleIndxr = new StringIndexer().setInputCol("RoofStyle")
                                        .setOutputCol("RoofStyleIdx")
                                        .setHandleInvalid("skip")

val heatingIndxr = new StringIndexer().setInputCol("Heating")
                                      .setOutputCol("HeatingIdx")
                                      .setHandleInvalid("skip")

val linearRegression = new LinearRegression().setLabelCol("SalePrice")
val assembler = new VectorAssembler().setInputCols(
                                     Array("LotArea", "RoofStyleIdx", "HeatingIdx",
                                           "LotArea", "BedroomAbvGr", "KitchenAbvGr", "GarageCars",
                                           "TotRmsAbvGrd", "YearBuilt", "TotalSF"))
                                     .setOutputCol("features")
// setup the pipeline
val pipeline = new Pipeline().setStages(Array(roofStyleIndxr, heatingIndxr, assembler, linearRegression))
// split the data into training and test pair
val Array(training, test) = skinny_house_data1.randomSplit(Array(0.8, 0.2))
// train the pipeline
val model = pipeline.fit(training)
// perform prediction
val predictions = model.transform(test)
val evaluator = new RegressionEvaluator().setLabelCol("SalePrice")
                                         .setPredictionCol("prediction")
                                         .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator()
evaluator.evaluate(predictions)
evaluator.getMetricName
